import os
from mcp.server.fastmcp import FastMCP
from langchain_openai import AzureChatOpenAI
from typing import List
import json
import ast
from openmindat import MineralsIMARetriever, GeomaterialRetriever
from pathlib import Path
import unicodedata
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

mcp = FastMCP("Mindat")

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
    
    normailized_mindat_mineral_list_path = _normalize_mindat_name(str(Path(mindat_output_dir, mindat_output_name + '.json')))
    
    return normailized_mindat_mineral_list_path
    

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
    
    normailized_mindat_mineral_list_path = _normalize_mindat_name(str(Path(rock_output_dir, rock_output_name + '.json')))
    
    return normailized_mindat_mineral_list_path

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
    for mineral in rock_list:
        key = mineral.lower()
        if key in variant_lookup:
            matched_results.append(variant_lookup[key])
        else:
            matched_results.append(f"unmatched_{mineral}")

    return matched_results

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
async def normalize_mindat_entry(input_dir: str, output_dir = 'data/processed') -> str:
    """
    Process all JSON files in the input directory:
    - Extract minerals from 'Mineralogy' field using LLM.
    - Replace 'Mineralogy' field with the extracted list.
    - Extract rocks from 'Rock_Types' field using LLM.
    - Replace 'Rock_Types' field with the extracted list.
    - Save results to the output directory using the same filenames.
    
    Args:
        input_dir: Path to the input folder containing .json files.
        output_dir: Path to the folder to write updated JSON files.
    
    Returns:
        A message summarizing how many files were processed successfully.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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
            raw_mineral_list = await extract_mineral_name_from_jsonfile(str(file), llm)
            matched_mineral_list = _match_minerals_to_mindat(raw_mineral_list)
            
            raw_rock_list = await extract_rock_name_from_jsonfile(str(file), llm)
            matched_rock_list = _match_rocks_to_mindat(raw_rock_list)
            
            with open(file, 'r', encoding='utf-8') as f:
                content = json.load(f)

            content['Mineralogy'] = matched_mineral_list
            content['Rock_Types'] = matched_rock_list

            output_file = output_path / file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)

            success_count += 1

        except Exception as e:
            failed_files.append(file.name)
            
    return_message = f"Processed {success_count} files successfully. Failed: {failed_files}." if failed_files else f"All {success_count} files processed successfully."

    return return_message + f" Results saved to {output_path}."

def check_synonyms():
    pass


if __name__ == "__main__":
    mcp.run(transport="stdio")
    
    # async def main():
    #     await normalize_mindat_entry('test/batch_test/final_json')

    # asyncio.run(main())
    
    # print(_match_minerals_to_mindat(['quartz', 'gibbsite', 'unobtainium']))
    # _update_mineral_list()