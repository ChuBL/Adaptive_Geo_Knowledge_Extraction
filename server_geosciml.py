import os
from mcp.server.fastmcp import FastMCP
import re
from typing import List, Tuple, Dict, Optional, Any
from langchain_openai import AzureChatOpenAI
import asyncio
import json
import ast
from dotenv import load_dotenv
from util.geosciml_vocab_updater import download_geosciml_vocabularies
from util.geosciml_vocab_parser import generate_vocab_descriptions, extract_ttl_members
from pathlib import Path
import glob
import concurrent.futures
import threading
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
mcp = FastMCP("GeosciML")

# Global configuration for output directory name
GLOBAL_OUTPUT_DIR_NAME = "4vocab_selection"

class GeosciMLLogManager:
    """Manages logging of GeosciML operations to geosciml_log_{timestamp}.json in source_{timestamp} directory."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.log_file_path = self._find_or_create_log_file()
        self.warning_log_file_path = self._find_or_create_warning_log_file()
        
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
        
        # Only create the source directory and vocab output subdirectory
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / GLOBAL_OUTPUT_DIR_NAME).mkdir(exist_ok=True)
        
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
                has_vocab_output = (self.project_root / GLOBAL_OUTPUT_DIR_NAME).exists()
                has_raw_ocr = (self.project_root / "raw_ocr").exists()
                
                if has_final_output or has_vocab_output or has_raw_ocr:
                    # We're likely at the right level (inside source directory)
                    source_dir = self.project_root
                    logger.info(f"Using current directory as source directory: {source_dir}")
                else:
                    # Only create new source structure as last resort
                    logger.warning(f"No source structure found. Creating new structure at {self.project_root}")
                    source_dir = self._create_source_structure()
        
        # Check for existing log files with timestamp pattern
        log_pattern = "geosciml_log_*.json"
        existing_logs = list(source_dir.glob(log_pattern))
        
        if existing_logs:
            # Use the first existing log file (sorted by name, which includes timestamp)
            log_file = sorted(existing_logs)[0]
            logger.info(f"Using existing geosciml log file: {log_file}")
        else:
            # Create new log file with timestamp
            log_file = source_dir / f"geosciml_log_{session_timestamp}.json"
            initial_data = {
                "log_info": {
                    "created_at": datetime.now().isoformat(),
                    "session_timestamp": session_timestamp,
                    "description": "GeosciML vocabulary matching and selection logs"
                },
                "operations": []
            }
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Created new geosciml log file: {log_file}")
        
        return log_file
    
    def _find_or_create_warning_log_file(self) -> Path:
        """Find existing warning log file or create new one in appropriate directory."""
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get the same source directory as main log
        source_dir = self.log_file_path.parent
        
        # Check for existing warning log files
        warning_log_pattern = "geosciml_warning_log_*.json"
        existing_warning_logs = list(source_dir.glob(warning_log_pattern))
        
        if existing_warning_logs:
            # Use the first existing warning log file
            warning_log_file = sorted(existing_warning_logs)[0]
            logger.info(f"Using existing geosciml warning log file: {warning_log_file}")
        else:
            # Create new warning log file with timestamp
            warning_log_file = source_dir / f"geosciml_warning_log_{session_timestamp}.json"
            initial_data = {
                "log_info": {
                    "created_at": datetime.now().isoformat(),
                    "session_timestamp": session_timestamp,
                    "description": "GeosciML warnings, timeouts, and validation errors"
                },
                "warnings": []
            }
            with open(warning_log_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Created new geosciml warning log file: {warning_log_file}")
        
        return warning_log_file
    
    def log_operation(self, 
                     filename: str,
                     operation_type: str,  # "vocabulary_file_selection", "vocabulary_term_selection", "file_processing"
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
    
    def log_warning(self,
                   filename: str,
                   warning_type: str,  # "file_processing_timeout", "llm_timeout", "validation_error"
                   start_time: datetime,
                   duration: float,
                   details: Dict[str, Any],
                   message: str) -> None:
        """Log a warning to the warning JSON file."""
        
        warning_record = {
            "timestamp": start_time.isoformat(),
            "filename": filename,
            "warning_type": warning_type,
            "duration_seconds": round(duration, 3),
            "details": details,
            "message": message
        }
        
        try:
            # Read existing warning log file
            with open(self.warning_log_file_path, 'r', encoding='utf-8') as f:
                warning_data = json.load(f)
            
            # Append new record
            warning_data["warnings"].append(warning_record)
            
            # Write back to file
            with open(self.warning_log_file_path, 'w', encoding='utf-8') as f:
                json.dump(warning_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Logged {warning_type} warning for {filename} to {self.warning_log_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to log warning: {e}")

# Global log manager instance
log_manager: Optional[GeosciMLLogManager] = None

def get_log_manager(project_root: Path) -> GeosciMLLogManager:
    """Get or create global log manager instance."""
    global log_manager
    if log_manager is None or log_manager.project_root != project_root:
        log_manager = GeosciMLLogManager(project_root)
    return log_manager

def determine_output_directory(input_dir: str) -> Path:
    """
    Create output directory at the same level as input directory.
    
    Args:
        input_dir: Path to the input folder containing .json files.
        
    Returns:
        Path to the output directory.
    """
    input_path = Path(input_dir)
    
    # Get parent directory of input path
    parent_dir = input_path.parent
    
    # Create output directory at the same level as input directory
    output_dir = parent_dir / GLOBAL_OUTPUT_DIR_NAME
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

async def _geosciml_initialize(vocab_path: str = "./data/vocabularies") -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Update the vocabulary list from the GeosciML database.
    This function downloads vocabularies and generates descriptions.
    Returns:
        Tuple containing the download results and description output path.
    """
    
    result_dict = download_geosciml_vocabularies(output_dir=vocab_path)
    
    # Check if descriptions file already exists
    descriptions_file_path = str(Path(vocab_path) / "_geosciml_descriptions.md")
    if os.path.exists(descriptions_file_path):
        print(f"GeosciML descriptions file already exists: {descriptions_file_path}")
        return result_dict, descriptions_file_path
    
    description_output_dir = await generate_vocab_descriptions(VOCAB_PATH = vocab_path,
                                      OUTPUT_PATH = descriptions_file_path)
    
    return result_dict, description_output_dir

# level 3, pick vocabulary in the ttl files based on geological description
async def pick_geosciml_vocabulary(
    input_text: str,
    geosciml_file_path: str,
    max_selections: int = 10,
    filename: str = "unknown",
    input_key: str = "unknown",
    log_mgr: Optional[GeosciMLLogManager] = None
) -> Dict[str, List[str]]:
    """
    Select the most relevant GeosciML vocabulary URI and members based on input geological description.
    """
    start_time = datetime.now()
    success = False
    selected_vocabularies = {}
    exception_info = None
    validation_passed = False
    validation_errors = []
    attempt_number = 0
    timeout_occurred = False
    time_out_limitation = 120.0
    
    try:
        # Return empty dict if input is empty
        if not input_text or not input_text.strip():
            success = True
            validation_passed = True
            return {}
        
        try:
            # Extract all collections from TTL file
            collections = extract_ttl_members(geosciml_file_path)
        except Exception as e:
            raise ValueError(f"Failed to read GeosciML vocabulary from {geosciml_file_path}: {e}")
        
        if not collections:
            success = True
            validation_passed = True
            return {}
        
        # Initialize LLM
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.3,
        )
        
        # Format all collections for better readability
        collections_formatted = ""
        for uri, members in collections.items():
            collections_formatted += f"\nURI: {uri}\n"
            collections_formatted += f"Members: {', '.join(members)}"
            collections_formatted += "\n"
        
        system_prompt = f"""You are a geological terminology expert specializing in GeosciML vocabulary classification.
            Your task: Analyze the geological description and identify the MOST relevant geological terms from the provided vocabulary, then return those terms with their corresponding URI containers.

            Available GeosciML collections:
            {collections_formatted}

            CRITICAL SELECTION CRITERIA:
            1. Focus on finding the most relevant TERMS that best match the geological description
            2. Terms must exist EXACTLY in the provided vocabulary - DO NOT create or modify any terms
            3. Maximum {max_selections} term selections total
            4. Quality over quantity - return fewer highly relevant terms rather than many marginally relevant ones
            5. Each selected term will be paired with its URI container for organization
            6. If validation errors are provided, carefully avoid selecting similar non-existent terms from the vocabulary

            SELECTION APPROACH:
            1. FIRST: Identify geological terms from the vocabulary that best match the input description semantically
            2. SECOND: Group these terms by their URI containers
            3. The URI is simply the organizational container - the terms are what matter

            EXAMPLES:
            - Input: "Paleozoic and Mesozoic" → Relevant terms: ["Paleozoic", "Mesozoic"] → Found in: Eras collection → {{"http://resource.geosciml.org/classifier/ics/ischart/Eras": ["Paleozoic", "Mesozoic"]}}
            - Input: "Jurassic period rocks" → Relevant term: ["Jurassic"] → Found in: Periods collection → {{"http://resource.geosciml.org/classifier/ics/ischart/Periods": ["Jurassic"]}}

            Output format: JSON object where keys are URI containers and values are lists of relevant terms found in those containers
            If no relevant terms found, return: {{}}

            IMPORTANT: Return ONLY the JSON object. Do NOT include any explanations, comments, code blocks, or additional text."""

        human_prompt = f"Geological description to analyze:\n\n{input_text}\n\nSelected relevant GeosciML vocabulary (JSON format):"
        
        # Reflection mechanism with up to 3 attempts, add timeout handling
        for attempt in range(3):
            attempt_number = attempt + 1
            try:
                # Key modification: Add 120-second timeout for LLM calls
                response = await asyncio.wait_for(
                    llm.ainvoke([
                        ("system", system_prompt),
                        ("human", human_prompt)
                    ]),
                    timeout=time_out_limitation  # 120-second timeout
                )
                result = str(response.content).strip()
                
                # Parse LLM response
                try:
                    if result.startswith('```json'):
                        result = result.replace('```json', '').replace('```', '').strip()
                    extracted = json.loads(result)
                except json.JSONDecodeError:
                    try:
                        extracted = ast.literal_eval(result)
                    except:
                        raise ValueError(f"Invalid JSON format in LLM response: {result}")
                
                # Validate the response format
                if not isinstance(extracted, dict):
                    raise ValueError(f"Expected dictionary, got {type(extracted)}")
                
                if not extracted:  # Empty dict is valid
                    selected_vocabularies = {}
                    validation_passed = True
                    success = True
                    break
                
                # Validate selections using reflection mechanism
                current_validation_errors = _validate_selections(extracted, collections)
                validation_errors.extend(current_validation_errors)
                
                if current_validation_errors:
                    validation_passed = False
                    # Add validation errors to the prompt for next attempt
                    error_message = "Previous selection had errors:\n" + "\n".join(current_validation_errors)
                    human_prompt += f"\n\n{error_message}\n\nPlease provide corrected selection:"
                    
                    # Log validation error warning
                    if log_mgr and attempt == 2:  # Last attempt
                        log_mgr.log_warning(
                            filename=filename,
                            warning_type="validation_error",
                            start_time=start_time,
                            duration=(datetime.now() - start_time).total_seconds(),
                            details={
                                "operation_type": "vocabulary_term_selection",
                                "input_key": input_key,
                                "ttl_file": os.path.basename(geosciml_file_path),
                                "validation_errors": validation_errors,
                                "attempt_number": attempt_number
                            },
                            message="Vocabulary selection validation failed"
                        )
                    continue
                else:
                    validation_passed = True
                    # validation_errors = []
                
                # Limit to max_selections and merge all relevant URIs
                validated_result = {}
                total_members = 0
                
                for uri, members in extracted.items():
                    if isinstance(members, list) and total_members < max_selections:
                        # Calculate how many members we can still add
                        remaining_slots = max_selections - total_members
                        selected_members = members[:remaining_slots]
                        
                        if selected_members:  # Only add if there are members to add
                            validated_result[uri] = selected_members
                            total_members += len(selected_members)
                
                selected_vocabularies = validated_result
                success = True
                break
                
            except asyncio.TimeoutError:
                timeout_occurred = True
                print(f"    LLM call timed out ({time_out_limitation}s) for attempt {attempt + 1}/3")
                
                if attempt == 2:  # Last attempt
                    # Log timeout warning
                    if log_mgr:
                        log_mgr.log_warning(
                            filename=filename,
                            warning_type="llm_timeout",
                            start_time=start_time,
                            duration=time_out_limitation,
                            details={
                                "operation_type": "vocabulary_term_selection",
                                "input_key": input_key,
                                "ttl_file": os.path.basename(geosciml_file_path),
                                "attempt_number": attempt_number,
                                "max_attempts": 3
                            },
                            message=f"LLM call timed out after {time_out_limitation} seconds on attempt {attempt_number}/3"
                        )
                
                
                    print("    All LLM attempts timed out, returning empty result")
                    exception_info = f"LLM timeout after {time_out_limitation} seconds"
                    break
                continue
            
            except Exception as e:
                if attempt == 2:  # Last attempt
                    exception_info = f"Failed to select GeosciML vocabulary after 3 attempts. Last error: {e}"
                    raise RuntimeError(exception_info)
                continue
        
    except Exception as e:
        exception_info = str(e)
        logger.error(f"Failed vocabulary term selection for {filename}: {e}")
        raise
    
    finally:
        # Log the operation if log_manager is provided
        if log_mgr:
            duration = (datetime.now() - start_time).total_seconds()
            details = {
                "input_key": input_key,
                "ttl_file": os.path.basename(geosciml_file_path),
                "selected_vocabularies": selected_vocabularies,
                "attempt_number": attempt_number,
                "validation_passed": validation_passed,
                "validation_errors": validation_errors
            }
            
            if timeout_occurred:
                details["timeout_occurred"] = True
            
            try:
                log_mgr.log_operation(
                    filename=filename,
                    operation_type="vocabulary_term_selection",
                    start_time=start_time,
                    duration=duration,
                    success=success,
                    details=details,
                    exception_info=exception_info
                )
            except Exception as log_error:
                logger.error(f"Failed to log vocabulary term selection: {log_error}")
    
    return selected_vocabularies


def _validate_selections(selections: Dict[str, List[str]], available_collections: Dict[str, List[str]]) -> List[str]:
    """
    Validate LLM selections against available collections using reflection mechanism.
    
    Args:
        selections: LLM selected URI and members
        available_collections: Available collections from TTL file
        
    Returns:
        List[str]: List of validation errors, empty if all valid
    """
    errors = []
    
    for selected_uri, selected_members in selections.items():
        # Check if URI exists
        if selected_uri not in available_collections:
            errors.append(f"URI '{selected_uri}' does not exist in the vocabulary file.")
            continue
        
        # Check if all selected members exist in the URI's member list
        available_members = available_collections[selected_uri]
        
        non_existent_members = []
        for member in selected_members:
            if member not in available_members:
                non_existent_members.append(member)
                # errors.append(f"Member '{member}' does not exist in URI '{selected_uri}'. Available members: {', '.join(available_members[:10])}{'...' if len(available_members) > 10 else ''}")

        if non_existent_members:
            errors.append(f"URI '{selected_uri}' has non-existent members: {', '.join(non_existent_members)}. Please check the vocabulary file for available members.")
    return errors


# level 2, pick vocabulary files based on geological description
async def pick_geosciml_vocabulary_files(
    input_text: str,
    descriptions_file_path: str,
    max_selections: int = 5,
    filename: str = "unknown",
    input_key: str = "unknown",
    log_mgr: Optional[GeosciMLLogManager] = None
) -> List[str]:
    """
    Select the most relevant GeosciML vocabulary TTL files based on input geological description.
    """
    start_time = datetime.now()
    success = False
    selected_files = []
    exception_info = None
    time_out_limitation = 120.0  # 120 seconds timeout for LLM calls
    
    try:
        try:
            # Read and parse the descriptions file
            vocabulary_descriptions = _read_vocabulary_descriptions(descriptions_file_path)
        except Exception as e:
            raise ValueError(f"Failed to read vocabulary descriptions from {descriptions_file_path}: {e}")
        
        if not vocabulary_descriptions:
            success = True
            return []
        
        # Initialize LLM
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.3,
        )
        
        # Format vocabulary descriptions for better readability
        vocabulary_formatted = "\n".join([
            f"- {filename}: {description}" 
            for filename, description in vocabulary_descriptions.items()
        ])
        
        system_prompt = f"""You are a geological terminology expert specializing in GeosciML vocabulary classification.

Your task: Analyze the geological description and select the most relevant GeosciML vocabulary files that would contain terms applicable to the content.

Available GeosciML vocabulary files ({len(vocabulary_descriptions)} total):
{vocabulary_formatted}

Selection criteria:
1. File names must exist exactly in the provided vocabulary list
2. Files must be relevant to the geological description content
3. Prioritize files that would contain the most specific and accurate terms for the description
4. ONLY select files that are truly relevant - do not feel obligated to reach the maximum of {max_selections} selections
5. Quality over quantity - it's better to return 1-2 highly relevant files than to include marginally relevant ones
6. Return filenames in order of relevance (most relevant first)
7. Only return the filename without the .ttl extension

Output format: Python list of strings with double quotes
Example: ["particle_aspect_ratio", "composition_category"]
If no relevant files found, return: []"""

        human_prompt = f"Geological description to analyze:\n\n{input_text}\n\nSelected relevant GeosciML vocabulary files:"
        
        last_exception = None
        
        for attempt in range(3):
            try:
                # Key modification: Add timeout for LLM calls
                response = await asyncio.wait_for(
                    llm.ainvoke([
                        ("system", system_prompt),
                        ("human", human_prompt)
                    ]),
                    timeout= time_out_limitation # 120 seconds timeout  
                )
                result = str(response.content).strip()
                
                try:
                    # Try to parse as JSON first
                    extracted = json.loads(result)
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval
                    extracted = ast.literal_eval(result)
                
                if isinstance(extracted, list) and all(isinstance(x, str) for x in extracted):
                    # Validate selections against available vocabulary files
                    available_files = [filename.replace('.ttl', '') for filename in vocabulary_descriptions.keys()]
                    valid_files = [filename for filename in extracted if filename in available_files]
                    
                    selected_files = valid_files[:max_selections]
                    success = True
                    break
                
            except asyncio.TimeoutError:
                print(f"    LLM call timed out ({time_out_limitation}s) for attempt {attempt + 1}/3")
                
                if attempt < 2:  # if not the last attempt
                    continue
                
                # Log timeout warning for the last attempt
                if log_mgr:
                    log_mgr.log_warning(
                        filename=filename,
                        warning_type="llm_timeout",
                        start_time=start_time,
                        duration=time_out_limitation,
                        details={
                            "operation_type": "vocabulary_file_selection",
                            "input_key": input_key,
                            "attempt_number": attempt + 1,
                            "max_attempts": 3
                        },
                        message=f"LLM call timed out after {time_out_limitation} seconds on attempt {attempt + 1}/3"
                    )
                
                last_exception = f"LLM timeout after {time_out_limitation} seconds"
                
                print("    All LLM attempts timed out, returning empty list")
                exception_info = last_exception
                break
            
            except Exception as e:
                last_exception = e
                continue
        
        if not success and last_exception:
            exception_info = str(last_exception)
            print(f"    Failed to select vocabulary files after 3 attempts. Last error: {last_exception}")
    
    except Exception as e:
        exception_info = str(e)
        logger.error(f"Failed vocabulary file selection for {filename}: {e}")
        raise
    
    finally:
        # Log the operation if log_manager is provided
        if log_mgr:
            duration = (datetime.now() - start_time).total_seconds()
            details = {
                "input_key": input_key,
                "selected_files": selected_files,
                "selected_count": len(selected_files)
            }
            
            try:
                log_mgr.log_operation(
                    filename=filename,
                    operation_type="vocabulary_file_selection",
                    start_time=start_time,
                    duration=duration,
                    success=success,
                    details=details,
                    exception_info=exception_info
                )
            except Exception as log_error:
                logger.error(f"Failed to log vocabulary file selection: {log_error}")
    
    return selected_files


def _read_vocabulary_descriptions(descriptions_file_path: str) -> Dict[str, str]:
    """
    Read and parse the GeosciML vocabulary descriptions markdown file.
    
    Args:
        descriptions_file_path (str): Path to the markdown descriptions file
        
    Returns:
        Dict[str, str]: Dictionary mapping filename to description
    """
    if not os.path.exists(descriptions_file_path):
        raise FileNotFoundError(f"Descriptions file not found: {descriptions_file_path}")
    
    try:
        with open(descriptions_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read descriptions file: {e}")
    
    descriptions = {}
    
    # Split content by markdown headers (##)
    sections = content.split('## ')[1:]  # Skip first empty element
    
    for section in sections:
        lines = section.strip().split('\n')
        if not lines:
            continue
            
        # First line should be the filename
        filename = lines[0].strip()
        
        # Look for the description line
        description = ""
        for line in lines[1:]:
            if line.startswith('**Description:**'):
                description = line.replace('**Description:**', '').strip()
                break
        
        if filename and description:
            descriptions[filename] = description
    
    return descriptions


# level 1, process JSON files to extract geological vocabularies
async def process_single_json_file(
    json_file_path: str,
    vocab_dir: str,
    descriptions_file_path: str,
    output_dir: str,
    log_mgr: Optional[GeosciMLLogManager] = None
) -> Optional[str]:
    """
    Process a single JSON file for vocabulary selection.
    Add timeout handling, skip the file if processing takes too long
    """
    
    target_keys = [
        "Textures",
        "Age_Range", 
        "Depositional_Environment",
        "Tectonic_Settings",
        "Associated_Deposit_Types",
        "Texture_Structure",
        "Alteration",
        "Ore_Controls",
        "Weathering",
        "Geochemical_Signature"
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(json_file_path)
    
    # skip the existing files
    output_file_path = os.path.join(output_dir, filename)
    if os.path.exists(output_file_path):
        print(f"  Skipping {filename} - already exists in output directory")
        return output_file_path

    try:
        # Key modification: Add timeout control for entire file processing
        return await asyncio.wait_for(
            _process_single_json_file_core(
                json_file_path, vocab_dir, descriptions_file_path, 
                output_dir, target_keys, filename, log_mgr
            ),
            timeout=600.0  # 10-minute timeout, give enough time for single file processing
        )
        
    except asyncio.TimeoutError:
        print(f"  *** TIMEOUT: Skipping {filename} - processing took longer than 5 minutes ***")
        
        # Log file processing timeout warning
        if log_mgr:
            log_mgr.log_warning(
                filename=filename,
                warning_type="file_processing_timeout",
                start_time=datetime.now(),
                duration=600.0,
                details={
                    "timeout_limit": 600,
                    "processing_stage": "entire_file_processing"
                },
                message="File processing exceeded 5-minute timeout limit"
            )
        
        return None
    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        return None


async def process_json_directory(
    input_dir: str,
    vocab_dir: str,
    descriptions_file_path: str,
    output_dir: str,
    log_mgr: Optional[GeosciMLLogManager] = None
) -> List[str]:
    """
    Process all JSON files in a directory, extract geological vocabularies for specific keys,
    and save the results with vocabulary selections.
    
    Args:
        input_dir (str): Directory containing input JSON files
        vocab_dir (str): Directory containing TTL vocabulary files
        descriptions_file_path (str): Path to the markdown file containing vocabulary descriptions
        output_dir (str): Output directory for processed JSON files
        log_mgr (GeosciMLLogManager): Log manager for recording operations
        
    Returns:
        List[str]: List of successfully processed output file paths
    """
    
    # Find all JSON files in input directory
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return []
    
    print(f"Found {len(json_files)} JSON files to process")
    
    successful_outputs = []
    
    # Process each JSON file using the single file processor
    for json_file in json_files:
        try:
            output_path = await process_single_json_file(
                json_file_path=json_file,
                vocab_dir=vocab_dir,
                descriptions_file_path=descriptions_file_path,
                output_dir=output_dir,
                log_mgr=log_mgr
            )
            
            if output_path:
                successful_outputs.append(output_path)
                
        except Exception as e:
            filename = os.path.basename(json_file)
            print(f"  Failed to process {filename}: {e}")
            continue
        
    
    print(f"Processing complete. Successfully processed {len(successful_outputs)}/{len(json_files)} files.")
    print(f"Results saved to {output_dir}")
    
    return successful_outputs

async def _process_single_json_file_core(
    json_file_path: str,
    vocab_dir: str, 
    descriptions_file_path: str,
    output_dir: str,
    target_keys: List[str],
    filename: str,
    log_mgr: Optional[GeosciMLLogManager] = None
) -> str:
    """Core processing logic, wrapped by timeout"""
    
    start_time = datetime.now()
    success = False
    processed_keys = []
    successful_keys = []
    failed_keys = []
    total_vocabulary_selections = 0
    exception_info = None
    
    try:
        # Read the original JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  Processing {filename}...")
        
        # Process each target key
        for key in target_keys:
            processed_keys.append(key)
            
            if key in data:
                original_value = data[key]
                
                # Skip if value is empty or None
                if not original_value or (isinstance(original_value, str) and not original_value.strip()):
                    print(f"    Skipping empty key: {key}")
                    data[key] = {}
                    continue
                
                # Convert to string for processing
                input_text = str(original_value)
                
                try:
                    # Step 1: Find relevant TTL files (with timeout and logging)
                    relevant_ttl_files = await pick_geosciml_vocabulary_files(
                        input_text=input_text,
                        descriptions_file_path=descriptions_file_path,
                        max_selections=5,
                        filename=filename,
                        input_key=key,
                        log_mgr=log_mgr
                    )
                    
                    if not relevant_ttl_files:
                        print(f"    No relevant TTL files found for key: {key}")
                        data[key] = {}
                        continue
                    
                    # Step 2: Get vocabulary from each relevant TTL file (with timeout and logging)
                    combined_vocabulary = {}
                    
                    for ttl_filename in relevant_ttl_files:
                        ttl_file_path = os.path.join(vocab_dir, f"{ttl_filename}.ttl")
                        
                        if not os.path.exists(ttl_file_path):
                            print(f"    Warning: TTL file not found: {ttl_file_path}")
                            continue
                        
                        try:
                            vocabulary_result = await pick_geosciml_vocabulary(
                                input_text=input_text,
                                geosciml_file_path=ttl_file_path,
                                max_selections=10,
                                filename=filename,
                                input_key=key,
                                log_mgr=log_mgr
                            )
                            
                            # Merge results into combined vocabulary
                            combined_vocabulary.update(vocabulary_result)
                            
                        except Exception as e:
                            print(f"    Error processing {ttl_filename} for key {key}: {e}")
                            continue
                    
                    # Update the key with combined vocabulary
                    data[key] = combined_vocabulary
                    total_vocabulary_selections += len(combined_vocabulary)
                    successful_keys.append(key)
                    print(f"    ✓ Processed key: {key} ({len(combined_vocabulary)} URI collections)")
                    
                except Exception as e:
                    print(f"    Error processing key {key}: {e}")
                    data[key] = {}
                    failed_keys.append(key)
                    continue
            else:
                print(f"    Key not found in data: {key}")
        
        # Save the processed JSON file
        output_file_path = os.path.join(output_dir, filename)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved processed file: {output_file_path}")
        success = True
        
    except Exception as e:
        exception_info = str(e)
        logger.error(f"Failed file processing for {filename}: {e}")
        raise
    
    finally:
        # Log the file processing operation if log_manager is provided
        if log_mgr:
            duration = (datetime.now() - start_time).total_seconds()
            details = {
                "processed_keys": processed_keys,
                "successful_keys": successful_keys,
                "failed_keys": failed_keys,
                "total_vocabulary_selections": total_vocabulary_selections
            }
            
            try:
                log_mgr.log_operation(
                    filename=filename,
                    operation_type="file_processing",
                    start_time=start_time,
                    duration=duration,
                    success=success,
                    details=details,
                    exception_info=exception_info
                )
            except Exception as log_error:
                logger.error(f"Failed to log file processing: {log_error}")
    
    return output_file_path

@mcp.tool()
async def match_geosciml_vocabularies(input_dir: str):
    """
    Process geological JSON files and match GeosciML vocabularies to geological descriptions.
    
    Args:
        input_dir: Directory containing input JSON files with mindat-matched geological descriptions

    Returns:
        str: Summary of processing results
    """
    
    vocab_dir = "./data/vocabularies"  # Directory containing GeosciML TTL vocabulary files  
    
    # Use simplified path logic
    descriptions_file_path = str(Path(vocab_dir) / "_geosciml_descriptions.md")  # Path to vocabulary descriptions markdown file
    output_dir = str(determine_output_directory(input_dir))  # Output directory for processed files with vocabulary annotations
    project_root = determine_project_root(input_dir)  # Project root for logging
    
    # Initialize log manager
    log_mgr = get_log_manager(project_root)
    logger.info(f"Initialized GeosciML logging to: {log_mgr.log_file_path}")

    try:
        # Initialize GeosciML vocabularies if needed
        await _geosciml_initialize(vocab_dir)
        
        # Process all JSON files
        output_files = await process_json_directory(
            input_dir=input_dir,
            vocab_dir=vocab_dir,
            descriptions_file_path=descriptions_file_path,
            output_dir=output_dir,
            log_mgr=log_mgr
        )
        
        return f"Successfully processed {len(output_files)} geological JSON files. Results saved to {output_dir}. Logs saved to {log_mgr.log_file_path}"
        
    except Exception as e:
        return f"Error processing geological JSON files: {str(e)}"



if __name__ == "__main__":
    mcp.run(transport="stdio")

    # async def main():
    #     message = await pick_geosciml_vocabulary("Paleozoic and Mesozoic", 'test/geosci_test/test_ics.ttl')
    #     print(message)

    # asyncio.run(main())
    
    # # print(match_minerals_to_mindat(['quartz', 'gibbsite', 'unobtainium']))
    # print(_geosciml_initialize())

    # selected = asyncio.run(example_usage())
    # print("Selected vocabulary:", selected)
    # asyncio.run(match_geosciml_vocabularies('test/mini_test'))