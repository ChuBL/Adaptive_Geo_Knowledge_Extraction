import os
import json
import shutil
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import asyncio

mcp = FastMCP("Preprocessor")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Global configuration for output directories
class PathConfig:
    """Global configuration for output directory names and structure."""
    
    # Default directory names
    RAW_OCR_DIR = "raw_ocr"
    CANDIDATE_DIR = "1candidate_extractions"
    FINAL_OUTPUT_DIR = "2preprocessed_output"
    
    # File naming patterns
    CANDIDATE_FILE_SUFFIX = "_candidates.json"
    FINAL_FILE_SUFFIX = "_preprocessed.json"
    LOG_FILE_PREFIX = "preprocessor_log_"
    
    @classmethod
    def get_candidate_filename(cls, stem: str) -> str:
        """Generate candidate extraction filename."""
        return f"{stem}{cls.CANDIDATE_FILE_SUFFIX}"
    
    @classmethod
    def get_final_filename(cls, stem: str) -> str:
        """Generate final output filename."""
        return f"{stem}{cls.FINAL_FILE_SUFFIX}"
    
    @classmethod
    def get_log_filename(cls, timestamp: str) -> str:
        """Generate log filename."""
        return f"{cls.LOG_FILE_PREFIX}{timestamp}.json"



class EntryExtractionResponse(BaseModel):
    """Structured output for geological OCR entry extraction."""

    Model_Index: str = Field(alias="Model Index", description="Short identifier like '13b' representing the model index.")
    Model_Name: str = Field(alias="Model Name", description="Name of the deposit model.")
    APPROXIMATE_SYNONYM: str = Field(alias="APPROXIMATE SYNONYM", description="Alternative names for the model.")
    DESCRIPTION: str = Field(description="General description of the deposit model.")
    GENERAL_REFERENCE: str = Field(alias="GENERAL REFERENCE", description="Cited references relevant to the model.")
    Rock_Types: str = Field(alias="Rock Types", description="Host or associated rock types.")
    Textures: str = Field(description="Observed or inferred textural information.")
    Age_Range: str = Field(alias="Age Range", description="Geological time range of deposit formation.")
    Depositional_Environment: str = Field(alias="Depositional Environment", description="Depositional setting of the deposit.")
    Tectonic_Settings: str = Field(alias="Tectonic Setting(s)", description="Tectonic environment of formation.")
    Associated_Deposit_Types: str = Field(alias="Associated Deposit Types", description="Geologically related deposit types.")
    Mineralogy: str = Field(description="Mineral content of the deposit.")
    Texture_Structure: str = Field(alias="Texture/Structure", description="Structural or morphological features.")
    Alteration: str = Field(description="Associated alteration patterns.")
    Ore_Controls: str = Field(alias="Ore Controls", description="Structural or stratigraphic controls on ore.")
    Weathering: str = Field(description="Weathering characteristics or products.")
    Geochemical_Signature: str = Field(alias="Geochemical Signature", description="Typical geochemical anomalies or indicators.")
    EXAMPLES: str = Field(description="Examples of deposits of this type.")
    COMMENTS: str = Field(description="Additional commentary.")
    DEPOSITS: str = Field(description="Deposit list or locality examples.")
    ai_modification_log: List[str] = Field(description="Log of OCR corrections and assumptions.")

    class Config:
        populate_by_name = True

class LogManager:
    """Manages logging of AI calls to preprocessor_log_{timestamp}.json in source_{timestamp} directory."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.log_file_path = self._find_or_create_log_file()
        
    def _find_source_directory(self) -> Optional[Path]:
        """Find source_{timestamp} directory in project root."""
        if not self.project_root.exists():
            return None
            
        # Look for directories matching source_{timestamp} pattern
        for item in self.project_root.iterdir():
            if item.is_dir() and item.name.startswith("source_"):
                logger.info(f"Found existing source directory: {item}")
                return item
        return None
    
    def _find_or_create_log_file(self) -> Path:
        """Find existing log file or create new one in source_{timestamp} directory."""
        # Generate timestamp for this session
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # First check if we're already inside a source_{timestamp} directory
        if self.project_root.name.startswith("source_"):
            # We're already in the source directory, use it directly
            source_dir = self.project_root
            logger.info(f"Using current source directory: {source_dir}")
        else:
            # Look for existing source_{timestamp} directory in project root
            source_dir = self._find_source_directory()
            
            if source_dir is None:
                # No source directory found, save in same level as raw_ocr
                # Check if project_root contains raw_ocr (meaning we're at the right level)
                raw_ocr_dir = self.project_root / PathConfig.RAW_OCR_DIR
                if raw_ocr_dir.exists():
                    # We're at the right level (inside source_{timestamp} directory)
                    source_dir = self.project_root
                    logger.info(f"Saving log in source directory (same level as {PathConfig.RAW_OCR_DIR}): {source_dir}")
                else:
                    # This shouldn't happen in normal workflow, but handle gracefully
                    logger.warning(f"No {PathConfig.RAW_OCR_DIR} directory found in {self.project_root}. Expected to be in source_{{timestamp}} directory.")
                    source_dir = self.project_root
        
        # Check for existing log files with timestamp pattern
        log_pattern = f"{PathConfig.LOG_FILE_PREFIX}*.json"
        existing_logs = list(source_dir.glob(log_pattern))
        
        if existing_logs:
            # Use the first existing log file (sorted by name, which includes timestamp)
            log_file = sorted(existing_logs)[0]
            logger.info(f"Using existing log file: {log_file}")
        else:
            # Create new log file with timestamp
            log_file = source_dir / PathConfig.get_log_filename(session_timestamp)
            initial_data = {
                "log_info": {
                    "created_at": datetime.now().isoformat(),
                    "session_timestamp": session_timestamp,
                    "description": "AI call logs for geological OCR preprocessing"
                },
                "ai_calls": []
            }
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Created new log file: {log_file}")
        
        return log_file
    
    def log_ai_call(self, 
                   filename: str,
                   operation_name: str,
                   start_time: datetime,
                   duration: float,
                   input_preview: str,
                   output_preview: str,
                   retry_count: int,
                   success: bool,
                   exception_info: Optional[str] = None) -> None:
        """Log a single AI call to the JSON file."""
        
        call_record = {
            "timestamp": start_time.isoformat(),
            "filename": filename,
            "operation": operation_name,
            "duration_seconds": round(duration, 3),
            "input_preview": input_preview[:100] + "..." if len(input_preview) > 100 else input_preview,
            "output_preview": output_preview[:100] + "..." if len(output_preview) > 100 else output_preview,
            "retry_count": retry_count,
            "success": success,
            "exception": exception_info
        }
        
        try:
            # Read existing log file
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # Append new record
            log_data["ai_calls"].append(call_record)
            
            # Write back to file
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Logged AI call for {filename} to {self.log_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to log AI call: {e}")

class AICallConfig:
    """Configuration class for AI model calls."""
    
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0
    
    # Model parameters for different task types
    EXTRACTION_CONFIG = {
        "temperature": 0.2,
        "max_retries": 3,
        "timeout": 120
    }
    
    COMPARISON_CONFIG = {
        "temperature": 0.1,
        "max_retries": 3,
        "timeout": 60
    }

class AIService:
    """Centralized AI service for consistent model interactions."""
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.parser = PydanticOutputParser(pydantic_object=EntryExtractionResponse)
        self.current_filename = "unknown"  # Track current file being processed
        
    def set_current_filename(self, filename: str) -> None:
        """Set the current filename being processed for logging."""
        self.current_filename = filename
        
    def _initialize_llm(self) -> AzureChatOpenAI:
        """Initialize the Azure OpenAI client with proper error handling."""
        try:
            required_env_vars = [
                "AZURE_DEPLOYMENT_NAME",
                "AZURE_OPENAI_API_VERSION", 
                "AZURE_OPENAI_API_ENDPOINT",
                "AZURE_OPENAI_API_KEY"
            ]
            
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {missing_vars}")
            
            return AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                temperature=AICallConfig.DEFAULT_TEMPERATURE,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    async def call_with_retry(
        self, 
        system_prompt: str, 
        user_prompt: str,
        config: Dict = None,
        parse_output: bool = False,
        operation_name: str = "AI_CALL",
        project_root: Optional[Path] = None
    ) -> Any:
        """
        Standardized method for making AI calls with retry logic and logging.
        
        Args:
            system_prompt: System message for the AI
            user_prompt: User message for the AI
            config: Configuration dict with temperature, max_retries, etc.
            parse_output: Whether to parse output with PydanticOutputParser
            operation_name: Name of the operation for logging
            project_root: Project root path for logging
            
        Returns:
            AI response as string or parsed object
        """
        if config is None:
            config = {"max_retries": AICallConfig.DEFAULT_MAX_RETRIES}
        
        max_retries = config.get("max_retries", AICallConfig.DEFAULT_MAX_RETRIES)
        temperature = config.get("temperature", self.llm.temperature)
        
        # Initialize logging variables
        start_time = datetime.now()
        last_exception = None
        retry_count = 0
        success = False
        output_preview = ""
        exception_info = None
        
        # Temporarily adjust temperature if specified
        original_temp = self.llm.temperature
        if temperature != original_temp:
            self.llm.temperature = temperature
        
        try:
            for attempt in range(max_retries):
                try:
                    logger.info(f"{operation_name}: Attempt {attempt + 1}/{max_retries}")
                    
                    # Enhance user prompt with retry context if needed
                    enhanced_user_prompt = user_prompt
                    if last_exception and attempt > 0:
                        enhanced_user_prompt += f"\n\nPrevious attempt failed with: {str(last_exception)}"
                    
                    # Make the AI call
                    response = await self.llm.ainvoke([
                        ("system", system_prompt),
                        ("human", enhanced_user_prompt)
                    ])
                    
                    # Parse output if requested
                    if parse_output:
                        try:
                            parsed_result = self.parser.invoke(response)
                            result = parsed_result.model_dump()
                            output_preview = str(result)
                        except OutputParserException as parse_error:
                            logger.warning(f"{operation_name}: Parse error on attempt {attempt + 1}: {parse_error}")
                            if attempt < max_retries - 1:
                                last_exception = parse_error
                                retry_count = attempt + 1
                                continue
                            else:
                                raise
                    else:
                        result = str(response.content).strip()
                        output_preview = result
                    
                    logger.info(f"{operation_name}: Successful on attempt {attempt + 1}")
                    success = True
                    retry_count = attempt
                    break
                    
                except Exception as e:
                    last_exception = e
                    retry_count = attempt + 1
                    logger.warning(f"{operation_name}: Attempt {attempt + 1} failed: {e}")
                    
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.error(f"{operation_name}: All {max_retries} attempts failed. Last error: {e}")
                        exception_info = str(e)
                        raise RuntimeError(f"{operation_name} failed after {max_retries} attempts. Last error: {e}")
            
            return result
        
        finally:
            # Restore original temperature
            if temperature != original_temp:
                self.llm.temperature = original_temp
            
            # Log the AI call if project_root is provided
            if project_root:
                duration = (datetime.now() - start_time).total_seconds()
                input_preview = f"System: {system_prompt[:50]}... | User: {user_prompt[:50]}..."
                
                try:
                    log_mgr = get_log_manager(project_root)
                    log_mgr.log_ai_call(
                        filename=self.current_filename,
                        operation_name=operation_name,
                        start_time=start_time,
                        duration=duration,
                        input_preview=input_preview,
                        output_preview=output_preview,
                        retry_count=retry_count,
                        success=success,
                        exception_info=exception_info
                    )
                except Exception as log_error:
                    logger.error(f"Failed to log AI call: {log_error}")

# Global log manager (will be initialized when needed)
log_manager: Optional[LogManager] = None

def get_log_manager(project_root: Path) -> LogManager:
    """Get or create global log manager instance."""
    global log_manager
    if log_manager is None or log_manager.project_root != project_root:
        log_manager = LogManager(project_root)
    return log_manager

# Global AI service instance
ai_service = AIService()

entry_extraction_prompt = """
    This tool processes OCR-extracted raw text from a geological descriptive document and extracts structured keyword entries and their associated content.
    Return your response as **pure JSON**, without any formatting or markdown. 
    Do not include triple backticks or code block formatting.

    The tool must identify a predefined set of field labels (see list below) and extract the text immediately associated with each. These field labels are mandatory: the output must include all of them, regardless of whether they are found in the input. If a label is not present in the text, or if no content follows it, an empty string ("") must be recorded as its value.

    Extraction rules:
    - The associated content begins immediately after the matched label and continues until the next recognized label or the end of section.
    - If duplicate labels appear, retain only the first valid occurrence.
    - Ignore blocks clearly associated with figure captions, image references, or chart labels.
    - Common OCR errors should be corrected conservatively. This includes fixing misread characters in field labels (e.g., "HODEL" → "MODEL"), restoring broken words, and correcting geological terms. All such corrections must be recorded.
    - Every correction and assumption must be saved under the key `ai_modification_log` as a list of strings.
    - Special handling for `Model Index`: this field is expected to match a short identifier composed of 1–2 digits followed by an optional lowercase letter (e.g., "13b"), typically appearing near the top of the section.

    Expected output:
    A single Python dictionary with the following characteristics:
    - Each key is one of the field labels listed below, plus the additional key `ai_modification_log`.
    - Each value is the extracted text content associated with the field, or an empty string if not found.
    - The value for `ai_modification_log` is a list of textual notes about OCR corrections or heuristic assumptions.

    Mandatory field labels:
    [
        "Model_Index",
        "Model_Name",
        "APPROXIMATE_SYNONYM",
        "DESCRIPTION",
        "GENERAL_REFERENCE",
        "Rock_Types",
        "Textures",
        "Age_Range",
        "Depositional_Environment",
        "Tectonic_Settings",
        "Associated_Deposit_Types",
        "Mineralogy",
        "Texture_Structure",
        "Alteration",
        "Ore_Controls",
        "Weathering",
        "Geochemical_Signature",
        "EXAMPLES",
        "COMMENTS",
        "DEPOSITS"
    ]
"""

extraction_comparison_prompt = """You are a senior economic geologist with expertise in mineral deposit modeling and geoscientific data curation. You are given three structured JSON-formatted versions of a descriptive model for carbonatite deposits, each derived from OCR-processed documents with subsequent AI-assisted correction.

    Your task is to compare these three versions and **select the single best version** based on the following expert criteria:

    1. **Terminological Accuracy** — Are geological terms, rock and mineral names, and technical expressions correctly rendered and spelled? Favor versions that accurately correct OCR artifacts (e.g., "monagite" → "monazite").

    2. **Geological Consistency** — Is the mineralogical, petrological, and tectonic content scientifically plausible and internally consistent from the standpoint of a trained geologist?

    3. **Language Clarity and Professional Tone** — Prefer precise, technical language without unnecessary simplification or hallucination. Avoid versions that introduce awkward or unclear phrasing.

    4. **Minimal Intervention Principle** — Favor versions that make only necessary and justifiable corrections to the original OCR output, preserving the authentic structure and content of the source.

    Each version already contains all required fields; your evaluation should focus only on **correctness and quality of the extracted content**.

    **Output only the single best version** (version1, version2, or version3). Do not rewrite, merge, or create a new version. Optionally, include a brief justification for your choice as a geologist.
"""

def determine_output_directories(input_path: Path) -> Dict[str, Path]:
    """
    Determine output directories based on input file path and existing structure using global PathConfig.
    
    Args:
        input_path: Path to the input .txt file
        
    Returns:
        Dictionary containing candidate_dir, final_dir, and project_root paths
    """
    logger.debug(f"Determining output directories for: {input_path}")
    
    # Check if input file is in a raw_ocr directory structure
    if input_path.parent.name == PathConfig.RAW_OCR_DIR:
        # Input is in raw_ocr, the project root should be the parent of raw_ocr
        project_root = input_path.parent.parent
        candidate_dir = project_root / PathConfig.CANDIDATE_DIR
        final_dir = project_root / PathConfig.FINAL_OUTPUT_DIR
        
        logger.info(f"Input file is in {PathConfig.RAW_OCR_DIR} directory. Project root: {project_root}")
        
        # Create directories if they don't exist (they should be at same level as raw_ocr)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            "candidate_dir": candidate_dir,
            "final_dir": final_dir,
            "project_root": project_root
        }
    
    # Check if we're already in a project root that contains raw_ocr
    current_dir = input_path.parent
    
    # First check the current directory
    raw_ocr_dir = current_dir / PathConfig.RAW_OCR_DIR
    if raw_ocr_dir.exists():
        # We're in project root
        project_root = current_dir
        candidate_dir = project_root / PathConfig.CANDIDATE_DIR
        final_dir = project_root / PathConfig.FINAL_OUTPUT_DIR
        
        logger.info(f"Found project root (contains {PathConfig.RAW_OCR_DIR}): {project_root}")
        
        # Create directories if they don't exist
        candidate_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            "candidate_dir": candidate_dir,
            "final_dir": final_dir,
            "project_root": project_root
        }
    
    # Traverse up to find project root
    while current_dir.parent != current_dir:  # Traverse up until we reach filesystem root
        current_dir = current_dir.parent
        raw_ocr_dir = current_dir / PathConfig.RAW_OCR_DIR
        
        # Check if this directory contains raw_ocr (indicates project root)
        if raw_ocr_dir.exists():
            project_root = current_dir
            candidate_dir = project_root / PathConfig.CANDIDATE_DIR
            final_dir = project_root / PathConfig.FINAL_OUTPUT_DIR
            
            logger.info(f"Found project root by traversing up: {project_root}")
            
            # Create directories if they don't exist
            candidate_dir.mkdir(parents=True, exist_ok=True)
            final_dir.mkdir(parents=True, exist_ok=True)
            
            return {
                "candidate_dir": candidate_dir,
                "final_dir": final_dir,
                "project_root": project_root
            }
    
    # Fallback: create directories in input file's parent directory
    parent_dir = input_path.parent
    candidate_dir = parent_dir / PathConfig.CANDIDATE_DIR
    final_dir = parent_dir / PathConfig.FINAL_OUTPUT_DIR
    
    # Create directories if they don't exist
    candidate_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"No existing project structure found. Created new directory structure in: {parent_dir}")
    return {
        "candidate_dir": candidate_dir,
        "final_dir": final_dir,
        "project_root": parent_dir
    }

async def generate_entry_versions(
    file_path: str,
    num_versions: int = 3
) -> Tuple[Path, str, Dict[str, Path]]:
    """
    Extract multiple structured versions from a .txt file using LLM, save raw JSON result, and return paths.

    Args:
        file_path: path to a .txt file.
        num_versions: number of versions to generate (default: 3).

    Returns:
        A tuple of (path to saved candidate JSON file, original raw text string, output directories dict).
    """
    logger.info(f"Generating {num_versions} entry versions for: {file_path}")
    
    input_path = Path(file_path)
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    if input_path.suffix.lower() != ".txt":
        raise ValueError(f"File must be a .txt file: {file_path}")

    # Set current filename in AI service for logging
    ai_service.set_current_filename(input_path.name)

    # Read input file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        logger.debug(f"Successfully read file: {file_path} ({len(raw_text)} characters)")
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise

    # Determine output directories based on file structure
    output_dirs = determine_output_directories(input_path)
    
    # Save candidate versions to candidate_extractions directory using PathConfig
    candidate_output_file = output_dirs["candidate_dir"] / PathConfig.get_candidate_filename(input_path.stem)

    all_versions: Dict[str, dict] = {}
    for i in range(num_versions):
        logger.info(f"Generating version {i+1}/{num_versions} for {input_path.name}")
        
        result_dict = await extract_single_structured_entry(
            raw_text, 
            version_number=i+1,
            total_versions=num_versions,
            project_root=output_dirs["project_root"]
        )
        all_versions[f"version{i+1}"] = result_dict

    # Save all versions to file
    try:
        with open(candidate_output_file, "w", encoding="utf-8") as f:
            json.dump(all_versions, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved candidate versions to: {candidate_output_file}")
    except Exception as e:
        logger.error(f"Failed to save candidate versions: {e}")
        raise

    return candidate_output_file, raw_text, output_dirs

async def extract_single_structured_entry(
    raw_text: str, 
    version_number: int = 1, 
    total_versions: int = 1,
    project_root: Optional[Path] = None
) -> dict:
    """
    Extract a single structured entry from raw text using LLM and return the parsed result.
    
    Args:
        raw_text: The raw OCR text to process
        version_number: Current version number for logging
        total_versions: Total number of versions being generated
        project_root: Project root path for logging
        
    Returns:
        Parsed dictionary result
    """
    operation_name = f"EXTRACTION_V{version_number}/{total_versions}"
    
    user_prompt = f"Please extract structured data from the following OCR text:\n\n{raw_text}"
    
    result = await ai_service.call_with_retry(
        system_prompt=entry_extraction_prompt,
        user_prompt=user_prompt,
        config=AICallConfig.EXTRACTION_CONFIG,
        parse_output=True,
        operation_name=operation_name,
        project_root=project_root
    )
    
    return result

async def select_best_entry_version(
    candidate_json_path: Path,
    input_stem: str,
    output_dirs: Dict[str, Path],
    raw_text: str
) -> str:
    """
    Use LLM to compare multiple structured versions and save the best one to final_output.
    
    Args:
        candidate_json_path: Path to the candidate versions JSON file
        input_stem: Stem name of the input file
        output_dirs: Dictionary containing output directory paths
        raw_text: Original raw OCR text for context
        
    Returns:
        Success message with final output path
    """
    logger.info(f"Selecting best version from: {candidate_json_path}")
    
    # Load candidate versions
    try:
        with open(candidate_json_path, 'r', encoding='utf-8') as f:
            versions = json.load(f)
        logger.debug(f"Loaded {len(versions)} candidate versions")
    except Exception as e:
        logger.error(f"Failed to load candidate versions: {e}")
        raise

    # Prepare comparison input
    comparison_input = f"==== EXTRACTED VERSIONS ====\n{json.dumps(versions, indent=2, ensure_ascii=False)}"
    system_prompt = extraction_comparison_prompt + "\n\n==== RAW OCR TEXT ====\n" + raw_text

    # Get LLM decision
    decision_response = await ai_service.call_with_retry(
        system_prompt=system_prompt,
        user_prompt=comparison_input,
        config=AICallConfig.COMPARISON_CONFIG,
        parse_output=False,
        operation_name="VERSION_COMPARISON",
        project_root=output_dirs.get("project_root")
    )

    # Parse decision
    decision_str = decision_response.lower().strip()
    selected_key = next((key for key in versions if key.lower() in decision_str), None)

    if not selected_key:
        logger.error(f"LLM failed to select valid version. Response: {decision_response}")
        raise ValueError(f"LLM failed to select a valid version. LLM response:\n{decision_str}")

    logger.info(f"Selected version: {selected_key}")

    # Save final result to final_output directory using PathConfig
    final_output_file = output_dirs["final_dir"] / PathConfig.get_final_filename(input_stem)
    try:
        with open(final_output_file, "w", encoding="utf-8") as f:
            json.dump(versions[selected_key], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved final result to: {final_output_file}")
    except Exception as e:
        logger.error(f"Failed to save final result: {e}")
        raise

    return f"Final version saved to: {final_output_file}"

def check_file_correspondence(project_root: Path) -> bool:
    """
    Check if files in raw_ocr and final_output directories correspond one-to-one.
    Returns True if they match, False otherwise. Passes on any exceptions.
    """
    try:
        raw_ocr_dir = project_root / PathConfig.RAW_OCR_DIR
        final_output_dir = project_root / PathConfig.FINAL_OUTPUT_DIR
        
        # Get file stems from both directories
        raw_files = {f.stem for f in raw_ocr_dir.glob("*.txt")}
        final_files = {f.stem.replace(PathConfig.FINAL_FILE_SUFFIX.replace('.json', ''), '') 
                      for f in final_output_dir.glob(f"*{PathConfig.FINAL_FILE_SUFFIX}")}
        
        return raw_files == final_files
    except:
        pass
        return False


@mcp.tool()
async def extract_entries_from_path(INPUT_PATH: str):
    """
    Extracts structured geological deposit model entries from a local file or directory.
    
    Input:
    - A string path to either a `.txt` file or a directory containing `.txt` files.
    
    Behavior:
    - For a file: generates multiple structured versions, selects the best one, and saves JSON files to appropriate directories.
    - For a directory: processes all top-level `.txt` files and saves structured outputs for each.
    
    Output:
    - Returns a string summary of the number of files processed, success count, failed cases, and the output directories.
    
    This tool adapts to existing OCR project structure or creates new directories as needed.
    """
    

    
    
    path_obj = Path(INPUT_PATH)
    
    if not path_obj.exists():
        error_msg = f"Path not found: {INPUT_PATH}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    parent_dir = path_obj.parent
    # Check if the json already processed in the parent directory
    if check_file_correspondence(parent_dir):
        return f"All files processed and saved in {parent_dir / PathConfig.FINAL_OUTPUT_DIR}. Please end this run and return the output dir to the supervisor agent to determine the next step."
    
    logger.info(f"Starting extraction process for: {INPUT_PATH}")
    

    if path_obj.is_file():
        if path_obj.suffix.lower() != ".txt":
            error_msg = "Only .txt files are supported for single-file processing."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            candidate_json_path, raw_text, output_dirs = await generate_entry_versions(str(path_obj))
            
            await select_best_entry_version(
                candidate_json_path=candidate_json_path,
                input_stem=path_obj.stem,
                output_dirs=output_dirs,
                raw_text=raw_text
            )
            
            success_msg = f"[Done] Processed: {path_obj.name}\n" \
                         f"→ Final result saved to: {output_dirs['final_dir'] / PathConfig.get_final_filename(path_obj.stem)}"
            logger.info(success_msg)
            return success_msg
            
        except Exception as e:
            error_msg = f"Failed to process file {path_obj.name}: {e}"
            logger.error(error_msg)
            raise
    
    elif path_obj.is_dir():
        txt_files = list(path_obj.glob("*.txt"))
        total_count = len(txt_files)
        
        if total_count == 0:
            warning_msg = f"No .txt files found in: {INPUT_PATH}"
            logger.warning(warning_msg)
            return warning_msg
        
        logger.info(f"Found {total_count} .txt files to process")
        
        success_count = 0
        failed_files = []
        all_output_dirs = set()  # Track unique output directories
        
        for txt_file in txt_files:
            try:
                logger.info(f"Processing file {success_count + len(failed_files) + 1}/{total_count}: {txt_file.name}")
                
                # Set current filename for logging
                ai_service.set_current_filename(txt_file.name)
                
                candidate_json_path, raw_text, output_dirs = await generate_entry_versions(str(txt_file))
                
                await select_best_entry_version(
                    candidate_json_path=candidate_json_path,
                    input_stem=txt_file.stem,
                    output_dirs=output_dirs,
                    raw_text=raw_text
                )
                success_count += 1
                all_output_dirs.add(str(output_dirs['candidate_dir'].parent))  # Add project root
                
            except Exception as e:
                logger.error(f"Failed to process {txt_file.name}: {e}")
                failed_files.append(txt_file.name)
        
        summary = f"→ Total files found: {total_count}\n" \
                  f"→ Successfully processed: {success_count}\n" \
                  f"→ Failed: {len(failed_files)}"
        
        if all_output_dirs:
            summary += f"\n→ Final output directories: {', '.join(str(Path(d) / PathConfig.FINAL_OUTPUT_DIR) for d in all_output_dirs)}"
        
        # if failed_files:
        #     summary += "\n→ Failed files:\n" + "\n".join(f"  - {f}" for f in failed_files)
        
        summary += "\nThe tool has completed the preprocessing task. Please end this run and return the output dir to the supervisor agent to determine the next step."
        logger.info(summary)
        return summary
    
    else:
        error_msg = "Provided PATH is neither a file nor a directory."
        logger.error(error_msg)
        raise ValueError(error_msg)


if __name__ == "__main__":
    mcp.run(transport="stdio")
    
       
    # async def main():
    #     msg = await extract_entries_from_path("./data/work_dir/source_20250731_210320/raw_ocr")
    #     print(msg)
        
    # asyncio.run(main())