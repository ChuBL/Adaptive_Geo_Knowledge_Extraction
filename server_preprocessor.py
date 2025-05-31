import os
import json
import shutil
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from typing import Dict
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("Preprocessor")



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

entry_extraction_prompt = """
    This tool processes OCR-extracted raw text from a geological descriptive document and extracts structured keyword entries and their associated content.
    Return your response as **pure JSON**, without any formatting or markdown. 
    Do not include triple backticks or code block formatting.

    The tool must identify a predefined set of field labels (see list below) and extract the text immediately associated with each. These field labels are mandatory: the output must include all of them, regardless of whether they are found in the input. If a label is not present in the text, or if no content follows it, an empty string ("") must be recorded as its value.

    Extraction rules:
    - The associated content begins immediately after the matched label and continues until the next recognized label or the end of section.
    - If duplicate labels appear, retain only the first valid occurrence.
    - Ignore blocks clearly associated with figure captions, image references, or chart labels.
    - Common OCR errors should be corrected conservatively. This includes fixing misread characters in field labels (e.g., “HODEL” → “MODEL”), restoring broken words, and correcting geological terms. All such corrections must be recorded.
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

extraction_comparison_prompt = """You are a senior economic geologist with expertise in mineral deposit modeling and geoscientific data curation. You are given three structured            JSON-formatted versions of a descriptive model for carbonatite deposits, each derived from OCR-processed documents with subsequent AI-assisted correction.

    Your task is to compare these three versions and **select the single best version** based on the following expert criteria:

    1. **Terminological Accuracy** — Are geological terms, rock and mineral names, and technical expressions correctly rendered and spelled? Favor versions that accurately correct OCR artifacts (e.g., “monagite” → “monazite”).

    2. **Geological Consistency** — Is the mineralogical, petrological, and tectonic content scientifically plausible and internally consistent from the standpoint of a trained geologist?

    3. **Language Clarity and Professional Tone** — Prefer precise, technical language without unnecessary simplification or hallucination. Avoid versions that introduce awkward or unclear phrasing.

    4. **Minimal Intervention Principle** — Favor versions that make only necessary and justifiable corrections to the original OCR output, preserving the authentic structure and content of the source.

    Each version already contains all required fields; your evaluation should focus only on **correctness and quality of the extracted content**.

    **Output only the single best version** (version1, version2, or version3). Do not rewrite, merge, or create a new version. Optionally, include a brief justification for your choice as a geologist.
"""

async def generate_entry_versions(
    FILE_PATH: str,
    llm: AzureChatOpenAI,
    parser: PydanticOutputParser,
    num_versions: int = 3
) -> tuple[Path, str]:
    """
    Extract multiple structured versions from a .txt file using LLM, save raw JSON result, and return both the JSON path and raw text.

    Args:
        FILE_PATH: path to a .txt file.
        llm: the language model to use.
        parser: output parser to parse LLM response.
        num_versions: number of versions to generate (default: 3).

    Returns:
        A tuple of (path to saved raw JSON file, original raw text string).
    """
    input_path = Path(FILE_PATH)
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"File not found: {FILE_PATH}")
    if input_path.suffix.lower() != ".txt":
        raise ValueError(f"File must be a .txt file: {FILE_PATH}")

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    raw_output_dir = input_path.parent / "raw_json"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_file = raw_output_dir / f"{input_path.stem}_raw.json"

    all_versions: Dict[str, dict] = {}
    for i in range(num_versions):
        result_dict = await extract_single_structured_entry(raw_text, llm, parser)
        all_versions[f"version{i+1}"] = result_dict

    with open(raw_output_file, "w", encoding="utf-8") as f:
        json.dump(all_versions, f, indent=2, ensure_ascii=False)

    return raw_output_file, raw_text

async def extract_single_structured_entry(raw_text: str, llm: AzureChatOpenAI, parser: PydanticOutputParser) -> dict:
    
    retry_tolerance = 3
    last_exception = None

    for attempt in range(retry_tolerance):
        try:
            prompt = f"Please extract:\n\n{raw_text}"
            if last_exception and attempt > 0:
                prompt += f"\n\nNote: Previous attempt #{attempt} failed with error: {str(last_exception)}"
            
            response = await llm.ainvoke([
                ("system", entry_extraction_prompt),
                ("human", prompt)
            ])
            parsed = parser.invoke(response)
            return parsed.model_dump()

        except Exception as e:
            last_exception = e
            if attempt < retry_tolerance - 1:
                # print(f"Extraction attempt {attempt + 1} failed: {e}, retrying...")
                pass
            else:
                raise RuntimeError(f"Extraction failed after {retry_tolerance} attempts. Last error: {e}")
            

async def select_best_entry_version(
    JSON_FILE_PATH: str,
    llm: AzureChatOpenAI,
    input_stem: str,
    final_dir: Path,
    raw_text: str
) -> str:
    """
    Use LLM to compare multiple structured versions (with access to original raw text) and save the best one.
    """
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        versions = json.load(f)

    comparison_input = f"==== EXTRACTED VERSIONS ====\n{json.dumps(versions, indent=2, ensure_ascii=False)}"
    retry_tolerance = 3
    last_exception = None
    decision = None

    for attempt in range(retry_tolerance):
        try:
            system_msg = extraction_comparison_prompt + "\n\n==== RAW OCR TEXT ====\n" + raw_text

            if last_exception and attempt > 0:
                comparison_input += f"\n\nNote: Previous attempt #{attempt} failed with error: {str(last_exception)}"

            decision = await llm.ainvoke([
                ("system", system_msg),
                ("human", comparison_input)
            ])
            break

        except Exception as e:
            last_exception = e
            if attempt < retry_tolerance - 1:
                continue
            else:
                raise RuntimeError(f"Comparison failed after {retry_tolerance} attempts. Last error: {e}")

    decision_str = str(decision).lower().strip()

    selected_key = next((key for key in versions if key.lower() in decision_str), None)

    if not selected_key:
        raise ValueError(f"LLM failed to select a valid version. LLM response:\n{decision_str}")

    final_dir.mkdir(parents=True, exist_ok=True)
    final_output_file = final_dir / f"{input_stem}_final.json"
    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(versions[selected_key], f, indent=2, ensure_ascii=False)

    return f"Final version saved to: {final_output_file}"

@mcp.tool()
async def extract_entries_from_path(PATH: str):
    """
        Extracts structured geological deposit model entries from a local file or directory.

        Input:
        - A string path to either a `.txt` file or a directory containing `.txt` files.

        Behavior:
        - For a file: generates multiple structured versions, selects the best one, and saves a JSON file.
        - For a directory: processes all top-level `.txt` files and saves structured outputs for each.

        Output:
        - Saves structured JSON files in 'raw_json' and 'final_json' folders.
        - Returns a string summary of the number of files processed, success count, and failed cases.

        This tool is suitable when a user provides OCR-based descriptive documents for automatic structuring.
    """
    path_obj = Path(PATH)
    if not path_obj.exists():
        raise FileNotFoundError(f"Path not found: {PATH}")

    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.3,
    )
    parser = PydanticOutputParser(pydantic_object=EntryExtractionResponse)

    if path_obj.is_file():
        if path_obj.suffix.lower() != ".txt":
            raise ValueError("Only .txt files are supported for single-file processing.")

        raw_json_path, raw_text = await generate_entry_versions(str(path_obj), llm, parser)

        await select_best_entry_version(
            JSON_FILE_PATH=str(raw_json_path),
            llm=llm,
            input_stem=path_obj.stem,
            final_dir=path_obj.parent / "final_json",
            raw_text=raw_text
        )

        return f"[Done] Processed: {path_obj.name}\n" \
               f"→ Raw JSON: {raw_json_path}\n" \
               f"→ Final JSON: {path_obj.parent / 'final_json' / f'{path_obj.stem}_final.json'}"

    elif path_obj.is_dir():
        txt_files = list(path_obj.glob("*.txt"))
        total_count = len(txt_files)
        if total_count == 0:
            return f"No .txt files found in: {PATH}"

        success_count = 0
        failed_files = []

        for txt_file in txt_files:
            # print(f"\n[Processing] {txt_file.name}")
            try:
                raw_json_path, raw_text = await generate_entry_versions(str(txt_file), llm, parser)

                await select_best_entry_version(
                    JSON_FILE_PATH=str(raw_json_path),
                    llm=llm,
                    input_stem=txt_file.stem,
                    final_dir=txt_file.parent / "final_json",
                    raw_text=raw_text
                )
                success_count += 1
            except Exception as e:
                # print(f"[Failed] {txt_file.name}: {e}")
                failed_files.append(txt_file.name)

        summary = f"[Done] Processed directory: {PATH}\n" \
                  f"→ Total files found: {total_count}\n" \
                  f"→ Successfully processed: {success_count}\n" \
                  f"→ Failed: {len(failed_files)}\n" \
                  f"→ Final output folder: {path_obj / 'final_json'}"

        if failed_files:
            summary += "\n→ Failed files:\n" + "\n".join(f"  - {f}" for f in failed_files)

        return summary

    else:
        raise ValueError("Provided PATH is neither a file nor a directory.")


if __name__ == "__main__":
    mcp.run(transport="stdio")