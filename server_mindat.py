import os
from mcp.server.fastmcp import FastMCP
from langchain_openai import AzureChatOpenAI
from typing import List, Dict, Optional, Any
import json
import ast
from openmindat import MineralsIMARetriever, GeomaterialRetriever
from pathlib import Path
import unicodedata
import asyncio
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Global configuration
MINDAT_OUTPUT_DIR = "3mindat_matched"

mcp = FastMCP("Mindat")

class MindatLogManager:
    """Manages logging of Mindat operations to mindat_log_{timestamp}.json in source_{timestamp} directory."""
    
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
    
    def _create_source_structure(self) -> Path:
        """Create new source_{timestamp} directory structure only when absolutely necessary."""
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_dir = self.project_root / f"source_{session_timestamp}"
        
        # Only create the source directory and mindat output subdirectory
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / MINDAT_OUTPUT_DIR).mkdir(exist_ok=True)
        
        logger.info(f"Created new source directory structure: {source_dir}")
        return source_dir
    
    def _find_or_create_log_file(self) -> Path:
        """Find existing log file or create new one in appropriate directory."""
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
                # Check if project_root contains any expected structure indicators
                has_final_output = (self.project_root / "final_output").exists()
                has_mindat_output = (self.project_root / MINDAT_OUTPUT_DIR).exists()
                has_raw_ocr = (self.project_root / "raw_ocr").exists()
                
                if has_final_output or has_mindat_output or has_raw_ocr:
                    # We're likely at the right level (inside source directory)
                    source_dir = self.project_root
                    logger.info(f"Using current directory as source directory: {source_dir}")
                else:
                    # Only create new source structure as last resort
                    logger.warning(f"No source structure found. Creating new structure at {self.project_root}")
                    source_dir = self._create_source_structure()
        
        # Check for existing log files with timestamp pattern
        log_pattern = "mindat_log_*.json"
        existing_logs = list(source_dir.glob(log_pattern))
        
        if existing_logs:
            # Use the first existing log file (sorted by name, which includes timestamp)
            log_file = sorted(existing_logs)[0]
            logger.info(f"Using existing mindat log file: {log_file}")
        else:
            # Create new log file with timestamp
            log_file = source_dir / f"mindat_log_{session_timestamp}.json"
            initial_data = {
                "log_info": {
                    "created_at": datetime.now().isoformat(),
                    "session_timestamp": session_timestamp,
                    "description": "Mindat matching and AI extraction logs"
                },
                "operations": []
            }
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Created new mindat log file: {log_file}")
        
        return log_file
    
    def log_operation(self, 
                     filename: str,
                     operation_type: str,  # "mineral_extraction", "rock_extraction", "mineral_matching", "rock_matching"
                     start_time: datetime,
                     duration: float,
                     success: bool,
                     details: Dict[str, Any],
                     exception_info: Optional[str] = None) -> None:
        """Log a single operation to the JSON file."""
        
        operation_record = {
            "timestamp": start_time.isoformat(),
            "filename": filename,
            "operation_type": operation_type,
            "duration_seconds": round(duration, 3),
            "success": success,
            "details": details,
            "exception": exception_info
        }
        
        try:
            # Read existing log file
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # Append new record
            log_data["operations"].append(operation_record)
            
            # Write back to file
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Logged {operation_type} operation for {filename} to {self.log_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to log operation: {e}")

class MindatAIService:
    """Centralized AI service for Mindat operations with logging."""
    
    def __init__(self):
        self.llm = self._initialize_llm()
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
                temperature=0.3,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    async def extract_with_logging(
        self,
        text: str,
        extraction_type: str,  # "mineral" or "rock"
        log_manager: Optional[MindatLogManager] = None
    ) -> List[str]:
        """
        Extract mineral or rock names with logging support.
        
        Args:
            text: Input text for extraction
            extraction_type: "mineral" or "rock"
            log_manager: Log manager instance for recording operations
            
        Returns:
            List of extracted mineral/rock names
        """
        start_time = datetime.now()
        operation_type = f"{extraction_type}_extraction"
        success = False
        extracted_items = []
        exception_info = None
        
        try:
            if extraction_type == "mineral":
                extracted_items = await extract_mineral_name_from_text(text, self.llm)
            elif extraction_type == "rock":
                extracted_items = await extract_rock_name_from_text(text, self.llm)
            else:
                raise ValueError(f"Invalid extraction_type: {extraction_type}")
            
            success = True
            
        except Exception as e:
            exception_info = str(e)
            logger.error(f"Failed {operation_type} for {self.current_filename}: {e}")
            raise
        
        finally:
            # Log the operation if log_manager is provided
            if log_manager:
                duration = (datetime.now() - start_time).total_seconds()
                details = {
                    "input_text_preview": text[:200] + "..." if len(text) > 200 else text,
                    "input_text_length": len(text),
                    "extracted_count": len(extracted_items),
                    "extracted_items": extracted_items
                }
                
                try:
                    log_manager.log_operation(
                        filename=self.current_filename,
                        operation_type=operation_type,
                        start_time=start_time,
                        duration=duration,
                        success=success,
                        details=details,
                        exception_info=exception_info
                    )
                except Exception as log_error:
                    logger.error(f"Failed to log {operation_type}: {log_error}")
        
        return extracted_items

# Global instances
log_manager: Optional[MindatLogManager] = None
ai_service = MindatAIService()

def get_log_manager(project_root: Path) -> MindatLogManager:
    """Get or create global log manager instance."""
    global log_manager
    if log_manager is None or log_manager.project_root != project_root:
        log_manager = MindatLogManager(project_root)
    return log_manager

def _update_mineral_list() -> str:
    """
    Update the mineral list from the Mindat database.
    This function should be implemented to fetch and update the mineral list.
    Returns:
        Path to the normalized Mindat mineral list JSON file.
    """
    mir = MineralsIMARetriever()
    mir.ima(1).fields('id,name').verbose(1)
    mindat_output_dir = 'data/mindat'
    mindat_output_name = 'mindat_ima_list'
    
    # Prevent overwriting if the normalized file already exists
    if os.path.exists(Path(mindat_output_dir, mindat_output_name + '_normalized.json')):
        return str(Path(mindat_output_dir, mindat_output_name + '_normalized.json'))
    
    mir.saveto(mindat_output_dir, mindat_output_name)
    
    normalized_mindat_mineral_list_path = _normalize_mindat_name(str(Path(mindat_output_dir, mindat_output_name + '.json')))
    
    return normalized_mindat_mineral_list_path
    

def _normalize_mindat_name(PATH: str) -> str:
    """
    Read a Mindat mineral list JSON file and, for each entry, add a 'name_variants' field.
    - 'name_variants' contains a list of unique, human-readable variations of the name:
        * The original name
        * The ASCII-only version (accents removed)
    - The original 'name' field is NOT modified.
    - Saves result to <original_filename>_normalized.json
    Returns: Path to the new JSON file as a string.
    """

    def remove_accents(s: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    with open(PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)['results']

    for entry in data:
        raw_name = entry["name"]
        ascii_name = remove_accents(raw_name)

        variants = list({raw_name, ascii_name})
        entry["name_variants"] = variants

    original_path = Path(PATH)
    normalized_path = original_path.with_name(f"{original_path.stem}_normalized.json")

    with open(normalized_path, 'w', encoding='utf-8') as file:
        json.dump({'results': data}, file, ensure_ascii=False, indent=2)

    return str(normalized_path)

def _match_minerals_to_mindat_with_logging(
    mineral_list: List[str], 
    filename: str,
    log_manager: Optional[MindatLogManager] = None
) -> List[str]:
    """
    Match a list of mineral names to the Mindat database with logging.

    For matched entries, returns 'mindat_{id}_{name}'.
    For unmatched, returns 'unmatched_{mineral}'.

    Args:
        mineral_list: A list of mineral names to match.
        filename: Current filename for logging.
        log_manager: Log manager instance for recording operations.

    Returns:
        A list of strings representing the match results.
    """
    start_time = datetime.now()
    success = False
    matched_results = []
    exception_info = None
    
    try:
        mindat_mineral_list_path = _update_mineral_list()

        with open(mindat_mineral_list_path, 'r', encoding='utf-8') as file:
            mindat_data = json.load(file)['results']

        variant_lookup = {}
        for entry in mindat_data:
            for variant in entry.get('name_variants', []):
                variant_lookup[variant.lower()] = f"mindat_{entry['id']}_{entry['name']}"

        matched_items = []
        unmatched_items = []
        
        for mineral in mineral_list:
            key = mineral.lower()
            if key in variant_lookup:
                result = variant_lookup[key]
                matched_results.append(result)
                matched_items.append(result)
            else:
                result = f"unmatched_{mineral}"
                matched_results.append(result)
                unmatched_items.append(mineral)

        success = True
        
    except Exception as e:
        exception_info = str(e)
        logger.error(f"Failed mineral matching for {filename}: {e}")
        raise
    
    finally:
        # Log the matching operation if log_manager is provided
        if log_manager:
            duration = (datetime.now() - start_time).total_seconds()
            details = {
                "input_minerals": mineral_list,
                "total_count": len(mineral_list),
                "matched_count": len([r for r in matched_results if r.startswith("mindat_")]),
                "unmatched_count": len([r for r in matched_results if r.startswith("unmatched_")]),
                "matched_items": [r for r in matched_results if r.startswith("mindat_")],
                "unmatched_items": [r.replace("unmatched_", "") for r in matched_results if r.startswith("unmatched_")]
            }
            
            try:
                log_manager.log_operation(
                    filename=filename,
                    operation_type="mineral_matching",
                    start_time=start_time,
                    duration=duration,
                    success=success,
                    details=details,
                    exception_info=exception_info
                )
            except Exception as log_error:
                logger.error(f"Failed to log mineral matching: {log_error}")

    return matched_results

def _match_minerals_to_mindat(mineral_list: List[str]) -> List[str]:
    """
    Match a list of mineral names to the Mindat database.

    For matched entries, returns 'mindat_{id}_{name}'.
    For unmatched, returns 'unmatched_{mineral}'.

    Args:
        mineral_list: A list of mineral names to match.

    Returns:
        A list of strings representing the match results.
    """
    mindat_mineral_list_path = _update_mineral_list()

    with open(mindat_mineral_list_path, 'r', encoding='utf-8') as file:
        mindat_data = json.load(file)['results']

    variant_lookup = {}
    for entry in mindat_data:
        for variant in entry.get('name_variants', []):
            variant_lookup[variant.lower()] = f"mindat_{entry['id']}_{entry['name']}"

    matched_results = []
    for mineral in mineral_list:
        key = mineral.lower()
        if key in variant_lookup:
            matched_results.append(variant_lookup[key])
        else:
            matched_results.append(f"unmatched_{mineral}")

    return matched_results

def _update_rock_list() -> str:
    """
    Update the rock list from the Mindat database.
    This function should be implemented to fetch and update the rock list.
    Returns:
        Path to the normalized Mindat mineral list JSON file.
    """
    gr = GeomaterialRetriever()
    # 0 - mineral, 1 - synonym, 2 - variety, 3 - mixture, 4 - series, 5 - grouplist, 6 - polytype, 7 - rock, 8 - commodity 
    gr._params.update({
                'entrytype': '7'
            })
    gr.fields('id,name,entrytype,entrytype_text').verbose(0)
    
    rock_output_dir = 'data/mindat'
    rock_output_name = 'mindat_rock_list'
    
    # Prevent overwriting if the normalized file already exists
    if os.path.exists(Path(rock_output_dir, rock_output_name + '_normalized.json')):
        return str(Path(rock_output_dir, rock_output_name + '_normalized.json'))
    
    gr.saveto(rock_output_dir, rock_output_name)
    
    normalized_mindat_rock_list_path = _normalize_mindat_name(str(Path(rock_output_dir, rock_output_name + '.json')))
    
    return normalized_mindat_rock_list_path

def _match_rocks_to_mindat_with_logging(
    rock_list: List[str], 
    filename: str,
    log_manager: Optional[MindatLogManager] = None
) -> List[str]:
    """
    Match a list of rock names to the Mindat database with logging.
    For matched entries, returns 'mindat_{id}_{name}'.
    For unmatched, returns 'unmatched_{rock}'.
    
    Args:
        rock_list: A list of rock names to match.
        filename: Current filename for logging.
        log_manager: Log manager instance for recording operations.
        
    Returns:
        A list of strings representing the match results.
    """
    start_time = datetime.now()
    success = False
    matched_results = []
    exception_info = None
    
    try:
        mindat_rock_list_path = _update_rock_list()

        with open(mindat_rock_list_path, 'r', encoding='utf-8') as file:
            mindat_data = json.load(file)['results']

        variant_lookup = {}
        for entry in mindat_data:
            for variant in entry.get('name_variants', []):
                variant_lookup[variant.lower()] = f"mindat_{entry['id']}_{entry['name']}"

        matched_items = []
        unmatched_items = []
        
        for rock in rock_list:
            key = rock.lower()
            if key in variant_lookup:
                result = variant_lookup[key]
                matched_results.append(result)
                matched_items.append(result)
            else:
                result = f"unmatched_{rock}"
                matched_results.append(result)
                unmatched_items.append(rock)

        success = True
        
    except Exception as e:
        exception_info = str(e)
        logger.error(f"Failed rock matching for {filename}: {e}")
        raise
    
    finally:
        # Log the matching operation if log_manager is provided
        if log_manager:
            duration = (datetime.now() - start_time).total_seconds()
            details = {
                "input_rocks": rock_list,
                "total_count": len(rock_list),
                "matched_count": len([r for r in matched_results if r.startswith("mindat_")]),
                "unmatched_count": len([r for r in matched_results if r.startswith("unmatched_")]),
                "matched_items": [r for r in matched_results if r.startswith("mindat_")],
                "unmatched_items": [r.replace("unmatched_", "") for r in matched_results if r.startswith("unmatched_")]
            }
            
            try:
                log_manager.log_operation(
                    filename=filename,
                    operation_type="rock_matching",
                    start_time=start_time,
                    duration=duration,
                    success=success,
                    details=details,
                    exception_info=exception_info
                )
            except Exception as log_error:
                logger.error(f"Failed to log rock matching: {log_error}")

    return matched_results

def _match_rocks_to_mindat(rock_list: List[str]) -> List[str]:
    """
    Match a list of rock names to the Mindat database.
    For matched entries, returns 'mindat_{id}_{name}'.
    For unmatched, returns 'unmatched_{rock}'.
    Args:
        rock_list: A list of rock names to match.
    Returns:
        A list of strings representing the match results.
    """
    mindat_rock_list_path = _update_rock_list()

    with open(mindat_rock_list_path, 'r', encoding='utf-8') as file:
        mindat_data = json.load(file)['results']

    variant_lookup = {}
    for entry in mindat_data:
        for variant in entry.get('name_variants', []):
            variant_lookup[variant.lower()] = f"mindat_{entry['id']}_{entry['name']}"

    matched_results = []
    for rock in rock_list:
        key = rock.lower()
        if key in variant_lookup:
            matched_results.append(variant_lookup[key])
        else:
            matched_results.append(f"unmatched_{rock}")

    return matched_results

def determine_output_directory(input_dir: str) -> Path:
    """
    Determine output directory based on input directory path.
    Look for source_{timestamp} structure and create mindat output directory within it.
    
    Args:
        input_dir: Path to the input folder containing .json files.
        
    Returns:
        Path to the output directory.
    """
    input_path = Path(input_dir)
    
    # Check if input directory is within a source_{timestamp} structure
    current_dir = input_path
    while current_dir.parent != current_dir:  # Traverse up until we reach filesystem root
        if current_dir.name.startswith("source_"):
            # Found source directory, create output folder within it
            output_dir = current_dir / MINDAT_OUTPUT_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
        current_dir = current_dir.parent
    
    # Fallback: check if we're already in source directory
    if input_path.name.startswith("source_") or (input_path / "final_output").exists():
        # We're likely in the source directory
        output_dir = input_path / MINDAT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    # Ultimate fallback: create output folder in input directory
    output_dir = input_path / MINDAT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def determine_project_root(input_dir: str) -> Path:
    """
    Determine project root based on input directory path.
    Look for source_{timestamp} structure.
    
    Args:
        input_dir: Path to the input folder containing .json files.
        
    Returns:
        Path to the project root directory (source_{timestamp} directory).
    """
    input_path = Path(input_dir)
    
    # Traverse up the directory tree to find source_{timestamp} directory
    current_dir = input_path
    while current_dir.parent != current_dir:  # Traverse up until we reach filesystem root
        if current_dir.name.startswith("source_"):
            # Found source directory, return it as project root
            logger.info(f"Found source directory as project root: {current_dir}")
            return current_dir
        current_dir = current_dir.parent
    
    # If no source directory found, check if input_path itself is a source directory
    if input_path.name.startswith("source_"):
        logger.info(f"Input path is source directory: {input_path}")
        return input_path
    
    # If we can't find source structure, return the input path
    # The log manager will handle creating structure if needed
    logger.warning(f"No source structure found, using input path as project root: {input_path}")
    return input_path

async def extract_mineral_name_from_jsonfile(
    FILE_PATH: str, llm: AzureChatOpenAI
) -> List[str]:
    """
    Use LLM to extract mineral names from a JSON file containing geological descriptions.

    Args:
        FILE_PATH: Path to the JSON file with geological descriptions.
        llm: An AzureChatOpenAI instance for extraction.

    Returns:
        A list of detected mineral names as strings.

    Raises:
        RuntimeError if extraction fails after 3 attempts.
    """
    with open(FILE_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)

    text = data.get("Mineralogy", "")

    return await extract_mineral_name_from_text(text, llm)

async def extract_rock_name_from_jsonfile(
    FILE_PATH: str, llm: AzureChatOpenAI
) -> List[str]:
    """
    Use LLM to extract rock names from a JSON file containing geological descriptions.

    Args:
        FILE_PATH: Path to the JSON file with geological descriptions.
        llm: An AzureChatOpenAI instance for extraction.

    Returns:
        A list of detected rock names as strings.

    Raises:
        RuntimeError if extraction fails after 3 attempts.
    """
    with open(FILE_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)

    text = data.get("Rock_Types", "")

    return await extract_rock_name_from_text(text, llm)

async def extract_mineral_name_from_text(text: str, llm: AzureChatOpenAI) -> List[str]:
    """
    Use LLM to extract mineral names from a geological description string.

    Args:
        text: A string describing mineralogical content, e.g., "Mainly gibbsite and mixture of gibbsite and boehmite..."
        llm: An AzureChatOpenAI instance for extraction.

    Returns:
        A list of detected mineral names as strings.

    Raises:
        RuntimeError if extraction fails after 3 attempts.
    """
    system_prompt = """You are a geological assistant. Your task is to extract a list of all valid mineral names from the given input.
Only include terms that are valid mineral species (e.g., quartz, hematite, anatase).
Ignore chemical formulas, adjectives, and non-mineral terms.
Output a Python list of strings using double quotes. Do not include explanations or formatting.
Example: ["quartz", "hematite", "gibbsite"]"""

    human_prompt = f"Input:\n{text}\n\nExtracted minerals:"

    last_exception = None

    for attempt in range(3):
        try:
            response = await llm.ainvoke([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            result = str(response.content).strip()

            # First try strict JSON loading
            try:
                extracted = json.loads(result)
            except json.JSONDecodeError:
                # Fallback: try safe Python evaluation
                extracted = ast.literal_eval(result)

            if isinstance(extracted, list) and all(isinstance(x, str) for x in extracted):
                return extracted

        except Exception as e:
            last_exception = e
            continue

    raise RuntimeError(f"Failed to extract minerals after 3 attempts. Last error: {last_exception}")

async def extract_rock_name_from_text(text: str, llm: AzureChatOpenAI) -> List[str]:
    """
    Use LLM to extract rock names from a geological description string.

    Args:
        text: A string describing lithological content, e.g., "Alluvial sand, gravel, and conglomerate indicative of rock types that host lode tin deposits...."
        llm: An AzureChatOpenAI instance for extraction.

    Returns:
        A list of detected rock names as strings.

    Raises:
        RuntimeError if extraction fails after 3 attempts.
    """
    system_prompt = """You are a geological assistant. Your task is to extract a list of all valid rock names from the given input.
Only include terms that are valid rock species (e.g., adinole, astridite, jasper).
Ignore chemical formulas, adjectives, and non-rock terms.
Output a Python list of strings using double quotes. Do not include explanations or formatting.
Example: ["quartz", "hematite", "gibbsite"]"""

    human_prompt = f"Input:\n{text}\n\nExtracted rocks:"

    last_exception = None

    for attempt in range(3):
        try:
            response = await llm.ainvoke([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            result = str(response.content).strip()

            # First try strict JSON loading
            try:
                extracted = json.loads(result)
            except json.JSONDecodeError:
                # Fallback: try safe Python evaluation
                extracted = ast.literal_eval(result)

            if isinstance(extracted, list) and all(isinstance(x, str) for x in extracted):
                return extracted

        except Exception as e:
            last_exception = e
            continue

    raise RuntimeError(f"Failed to extract minerals after 3 attempts. Last error: {last_exception}")



@mcp.tool()
async def normalize_mindat_entry(input_dir: str) -> str:
    """
    Process all JSON files in the input directory:
    - Extract minerals from 'Mineralogy' field using LLM.
    - Replace 'Mineralogy' field with the extracted list.
    - Extract rocks from 'Rock_Types' field using LLM.
    - Replace 'Rock_Types' field with the extracted list.
    - Save results to the output directory within the source structure.
    - Log all operations to mindat_log_{timestamp}.json
    
    Args:
        input_dir: Path to the input folder containing .json files.
    
    Returns:
        A message summarizing how many files were processed successfully.
    """
    input_path = Path(input_dir)
    output_path = determine_output_directory(input_dir)
    project_root = determine_project_root(input_dir)
    
    # Initialize log manager
    log_mgr = get_log_manager(project_root)
    logger.info(f"Initialized logging to: {log_mgr.log_file_path}")

    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.3,
    )

    success_count = 0
    failed_files = []

    for file in input_path.glob("*.json"):
        try:
            # Set current filename for AI service logging
            ai_service.set_current_filename(file.name)
            
            # Load file content to get original text for logging
            with open(file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # Extract minerals with logging
            mineralogy_text = content.get("Mineralogy", "")
            if mineralogy_text:
                raw_mineral_list = await ai_service.extract_with_logging(
                    mineralogy_text, "mineral", log_mgr
                )
                matched_mineral_list = _match_minerals_to_mindat_with_logging(
                    raw_mineral_list, file.name, log_mgr
                )
            else:
                raw_mineral_list = []
                matched_mineral_list = []
                # Log empty extraction
                start_time = datetime.now()
                log_mgr.log_operation(
                    filename=file.name,
                    operation_type="mineral_extraction",
                    start_time=start_time,
                    duration=0.0,
                    success=True,
                    details={
                        "input_text_preview": "",
                        "input_text_length": 0,
                        "extracted_count": 0,
                        "extracted_items": []
                    }
                )
                log_mgr.log_operation(
                    filename=file.name,
                    operation_type="mineral_matching",
                    start_time=start_time,
                    duration=0.0,
                    success=True,
                    details={
                        "input_minerals": [],
                        "total_count": 0,
                        "matched_count": 0,
                        "unmatched_count": 0,
                        "matched_items": [],
                        "unmatched_items": []
                    }
                )
            
            # Extract rocks with logging
            rock_types_text = content.get("Rock_Types", "")
            if rock_types_text:
                raw_rock_list = await ai_service.extract_with_logging(
                    rock_types_text, "rock", log_mgr
                )
                matched_rock_list = _match_rocks_to_mindat_with_logging(
                    raw_rock_list, file.name, log_mgr
                )
            else:
                raw_rock_list = []
                matched_rock_list = []
                # Log empty extraction
                start_time = datetime.now()
                log_mgr.log_operation(
                    filename=file.name,
                    operation_type="rock_extraction",
                    start_time=start_time,
                    duration=0.0,
                    success=True,
                    details={
                        "input_text_preview": "",
                        "input_text_length": 0,
                        "extracted_count": 0,
                        "extracted_items": []
                    }
                )
                log_mgr.log_operation(
                    filename=file.name,
                    operation_type="rock_matching",
                    start_time=start_time,
                    duration=0.0,
                    success=True,
                    details={
                        "input_rocks": [],
                        "total_count": 0,
                        "matched_count": 0,
                        "unmatched_count": 0,
                        "matched_items": [],
                        "unmatched_items": []
                    }
                )
            
            # Update content with matched results
            content['Mineralogy'] = matched_mineral_list
            content['Rock_Types'] = matched_rock_list

            # Save updated file
            output_file = output_path / file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)

            success_count += 1
            logger.info(f"Successfully processed {file.name}")

        except Exception as e:
            failed_files.append(file.name)
            logger.error(f"Failed to process {file.name}: {e}")
            
            # Log the failure
            try:
                log_mgr.log_operation(
                    filename=file.name,
                    operation_type="file_processing",
                    start_time=datetime.now(),
                    duration=0.0,
                    success=False,
                    details={"error_type": "file_processing_error"},
                    exception_info=str(e)
                )
            except Exception as log_error:
                logger.error(f"Failed to log error for {file.name}: {log_error}")
            
    return_message = f"Processed {success_count} files successfully. Failed: {failed_files}." if failed_files else f"All {success_count} files processed successfully."

    return return_message + f" Results saved to {output_path}. Log saved to {log_mgr.log_file_path}."

def check_synonyms():
    pass


if __name__ == "__main__":
    mcp.run(transport="stdio")
    
    # async def main():
    #     await normalize_mindat_entry('test/batch_test/final_json')

    # asyncio.run(main())
    
    # print(_match_minerals_to_mindat(['quartz', 'gibbsite', 'unobtainium']))
    # _update_mineral_list()