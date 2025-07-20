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

load_dotenv()
mcp = FastMCP("GeosciML")

async def _geosciml_initialize(vocab_path: str = "./data/vocabularies") -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Update the mineral list from the Mindat database.
    This function should be implemented to fetch and update the mineral list.
    Returns:
        Tuple containing the parsed output directory and a dictionary with download results.
    """
    
    result_dict = download_geosciml_vocabularies(output_dir=vocab_path)
    
    description_output_dir = await generate_vocab_descriptions(VOCAB_PATH = vocab_path,
                                      OUTPUT_PATH = str(Path(vocab_path, "geosciml_descriptions.md")))
    
    return result_dict, description_output_dir

# level 3, pick vocabulary in the ttl files based on geological description
async def pick_geosciml_vocabulary(
    input_text: str,
    geosciml_file_path: str,
    max_selections: int = 10
) -> Dict[str, List[str]]:
    """
    Select the most relevant GeosciML vocabulary URI and members based on input geological description.
    """
    # Return empty dict if input is empty
    if not input_text or not input_text.strip():
        return {}
    
    try:
        # Extract all collections from TTL file
        collections = extract_ttl_members(geosciml_file_path)
    except Exception as e:
        raise ValueError(f"Failed to read GeosciML vocabulary from {geosciml_file_path}: {e}")
    
    if not collections:
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
        try:
            # *** Key modification: Add 60-second timeout for LLM calls ***
            response = await asyncio.wait_for(
                llm.ainvoke([
                    ("system", system_prompt),
                    ("human", human_prompt)
                ]),
                timeout=60.0  # 60-second timeout
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
                return {}
            
            # Validate selections using reflection mechanism
            validation_errors = _validate_selections(extracted, collections)
            
            if validation_errors:
                # Add validation errors to the prompt for next attempt
                error_message = "Previous selection had errors:\n" + "\n".join(validation_errors)
                human_prompt += f"\n\n{error_message}\n\nPlease provide corrected selection:"
                continue
            
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
            
            return validated_result
            
        except asyncio.TimeoutError:
            print(f"    LLM call timed out (60s) for attempt {attempt + 1}/3")
            if attempt == 2:  # Last attempt
                print("    All LLM attempts timed out, returning empty result")
                return {}
            continue
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise RuntimeError(f"Failed to select GeosciML vocabulary after 3 attempts. Last error: {e}")
            continue
    
    return {}


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
        for member in selected_members:
            if member not in available_members:
                errors.append(f"Member '{member}' does not exist in URI '{selected_uri}'. Available members: {', '.join(available_members[:10])}{'...' if len(available_members) > 10 else ''}")
    
    return errors


# level 2, pick vocabulary files based on geological description
async def pick_geosciml_vocabulary_files(
    input_text: str,
    descriptions_file_path: str = "data/vocabularies/geosciml_descriptions.md",
    max_selections: int = 5
) -> List[str]:
    """
    Select the most relevant GeosciML vocabulary TTL files based on input geological description.
    """
    try:
        # Read and parse the descriptions file
        vocabulary_descriptions = _read_vocabulary_descriptions(descriptions_file_path)
    except Exception as e:
        raise ValueError(f"Failed to read vocabulary descriptions from {descriptions_file_path}: {e}")
    
    if not vocabulary_descriptions:
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
            # *** Key modification: Add 60-second timeout for LLM calls ***
            response = await asyncio.wait_for(
                llm.ainvoke([
                    ("system", system_prompt),
                    ("human", human_prompt)
                ]),
                timeout=60.0  # 60-second timeout
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
                
                return valid_files[:max_selections]
            
        except asyncio.TimeoutError:
            print(f"    LLM call timed out (60s) for attempt {attempt + 1}/3")
            last_exception = f"LLM timeout after 60 seconds"
            if attempt == 2:  # Last attempt
                print("    All LLM attempts timed out, returning empty list")
                return []
            continue
        except Exception as e:
            last_exception = e
            continue
    
    print(f"    Failed to select vocabulary files after 3 attempts. Last error: {last_exception}")
    return []


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
    output_dir: str = "data/vocab_selection"
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
        # *** Key modification: Add timeout control for entire file processing ***
        return await asyncio.wait_for(
            _process_single_json_file_core(
                json_file_path, vocab_dir, descriptions_file_path, 
                output_dir, target_keys, filename
            ),
            timeout=300.0  # 5-minute timeout, give enough time for single file processing
        )
        
    except asyncio.TimeoutError:
        print(f"  *** TIMEOUT: Skipping {filename} - processing took longer than 5 minutes ***")
        return None
    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        return None


async def process_json_directory(
    input_dir: str,
    vocab_dir: str,
    descriptions_file_path: str,
    output_dir: str = "data/vocab_selection"
) -> List[str]:
    """
    Process all JSON files in a directory, extract geological vocabularies for specific keys,
    and save the results with vocabulary selections.
    
    Args:
        input_dir (str): Directory containing input JSON files
        vocab_dir (str): Directory containing TTL vocabulary files
        descriptions_file_path (str): Path to the markdown file containing vocabulary descriptions
        output_dir (str): Output directory for processed JSON files
        
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
                output_dir=output_dir
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
    filename: str
) -> str:
    """Core processing logic, wrapped by timeout"""
    
    # Read the original JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Processing {filename}...")
    
    # Process each target key
    for key in target_keys:
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
                # Step 1: Find relevant TTL files (with timeout)
                relevant_ttl_files = await pick_geosciml_vocabulary_files(
                    input_text=input_text,
                    descriptions_file_path=descriptions_file_path,
                    max_selections=5
                )
                
                if not relevant_ttl_files:
                    print(f"    No relevant TTL files found for key: {key}")
                    data[key] = {}
                    continue
                
                # Step 2: Get vocabulary from each relevant TTL file (with timeout)
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
                            max_selections=10
                        )
                        
                        # Merge results into combined vocabulary
                        combined_vocabulary.update(vocabulary_result)
                        
                    except Exception as e:
                        print(f"    Error processing {ttl_filename} for key {key}: {e}")
                        continue
                
                # Update the key with combined vocabulary
                data[key] = combined_vocabulary
                print(f"    ✓ Processed key: {key} ({len(combined_vocabulary)} URI collections)")
                
            except Exception as e:
                print(f"    Error processing key {key}: {e}")
                data[key] = {}
                continue
        else:
            print(f"    Key not found in data: {key}")
    
    # Save the processed JSON file
    output_file_path = os.path.join(output_dir, filename)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved processed file: {output_file_path}")
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
    
    vocab_dir="data/vocabularies"  # Directory containing GeosciML TTL vocabulary files  
    descriptions_file_path="data/vocabularies/geosciml_descriptions.md"  # Path to vocabulary descriptions markdown file
    output_dir="data/vocab_selection"  # Output directory for processed files with vocabulary annotations

    try:
        # Initialize GeosciML vocabularies if needed
        await _geosciml_initialize(vocab_dir)
        
        # Process all JSON files
        output_files = await process_json_directory(
            input_dir=input_dir,
            vocab_dir=vocab_dir,
            descriptions_file_path=descriptions_file_path,
            output_dir=output_dir
        )
        
        return f"Successfully processed {len(output_files)} geological JSON files. Results saved to {output_dir}"
        
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
