import os
import tempfile
from typing import Optional
from pathlib import Path
from datetime import datetime
from pdf2image import convert_from_path
import pytesseract
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("OCR")

# Define base directory structure
BASE_OCR_DIR = Path("./data/work_dir")

def setup_pdf_directory_structure(pdf_name: str) -> dict:
    """
    Create standardized directory structure for the specified PDF
    
    Args:
        pdf_name: PDF filename (without extension) or full path
    
    Returns:
        Dictionary containing directory paths
    """
    # Extract just the filename if a full path is provided
    pdf_name = Path(pdf_name).stem
    
    pdf_dir = BASE_OCR_DIR / pdf_name
    
    dirs = {
        'pdf_root': pdf_dir,
        'raw_ocr': pdf_dir / 'raw_ocr',
        # 'candidate_extractions': pdf_dir / 'candidate_extractions', 
        # 'final_output': pdf_dir / 'final_output',
        'temp': pdf_dir / 'temp'
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def _process_single_pdf(file_path: Path, starting_page: Optional[int] = 0, ending_page: Optional[int] = None, show_preview: bool = True) -> dict:
    """
    Internal function to process a single PDF file.
    
    Args:
        file_path: Path to the PDF file
        starting_page: The index (0-based) of the first page to process
        ending_page: The index (0-based, exclusive) of the last page to process
        show_preview: Whether to include text preview in result
    
    Returns:
        dict: Processing result with status, message, and optional preview
    """
    if not file_path.exists():
        return {"status": "error", "message": f"File not found: {file_path}"}
    
    # Get PDF filename (without extension)
    pdf_name = file_path.stem
    
    # Setup standard directory structure
    dirs = setup_pdf_directory_structure(pdf_name)
    
    # Create output file path - save to raw_ocr directory
    output_txt_path = dirs['raw_ocr'] / f"{pdf_name}.txt"
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    extracted_text_str = ""
    
    try:
        pages = convert_from_path(str(file_path), dpi=300, fmt="png")
        custom_config = r'--oem 1 --psm 6 -l eng'
        
        processed_pages = 0
        for i, page in enumerate(pages):
            if starting_page is not None and i < starting_page:
                continue
            if ending_page is not None and i >= ending_page:
                break
            
            temp_img_path = os.path.join(temp_dir, f"page_{i}.png")
            page.save(temp_img_path)
            
            try:
                text = pytesseract.image_to_string(temp_img_path, config=custom_config)
            except Exception as e:
                text = f"[OCR failed on page {i + 1}: {str(e)}]"
            
            extracted_text_str += f"\n\n--- PAGE {i + 1} ---\n\n{text}"
            processed_pages += 1
        
        # Ensure proper UTF-8 encoding
        extracted_text_str = extracted_text_str.encode('utf-8', errors='replace').decode('utf-8')
        
        # Save extracted text to raw_ocr directory
        with open(output_txt_path, 'w', encoding='utf-8') as output_file:
            output_file.write(extracted_text_str)
        
        result = {
            "status": "success",
            "filename": file_path.name,
            "processed_pages": processed_pages,
            "output_path": str(output_txt_path)
        }
        
        if show_preview:
            preview_text = extracted_text_str[:1000]
            if len(extracted_text_str) > 1000:
                preview_text += "\n\n[... text truncated for preview ...]"
            result["preview"] = preview_text
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": f"Error during OCR processing: {str(e)}"}
    
    finally:
        # Clean up temporary files
        for fname in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, fname))
            except Exception:
                pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

@mcp.tool()
def single_ocr(FILE_PATH: str, STARTING_PAGE: Optional[int] = 0, ENDING_PAGE: Optional[int] = None) -> str:
    """
    Extract text from a PDF file using OCR and save results to standardized directory structure.
    
    Args:
        FILE_PATH (str): Path to the PDF file.
        STARTING_PAGE (int, optional): The index (0-based) of the first page to process.
        ENDING_PAGE (int, optional): The index (0-based, exclusive) of the last page to process.
    
    Returns:
        str: Processing summary with counts and output path.
    """
    file_path = Path(FILE_PATH)
    result = _process_single_pdf(file_path, STARTING_PAGE, ENDING_PAGE, show_preview=False)
    
    if result["status"] == "error":
        return f"Processed: 1\nSuccessful: 0\nError: {result['message']}"
    
    return f"Processed: 1\nSuccessful: 1\nOutput: {result['output_path']}"

@mcp.tool()
def multi_ocr(DIRECTORY_PATH: str, OVERWRITE: Optional[bool] = False) -> str:
    """
    Extract text from all PDF files in a directory using OCR and save to standardized directory structure.
    
    Args:
        DIRECTORY_PATH (str): Path to the directory containing PDF files.
        OVERWRITE (bool, optional): Whether to overwrite existing output files. Defaults to False.
    
    Returns:
        str: Processing summary with counts and raw ocr output directory.
    """
    
    directory_path = Path(DIRECTORY_PATH)
    if not directory_path.exists():
        return f"Processed: 0\nSuccessful: 0\nError: Directory not found: {DIRECTORY_PATH}"
    
    if not directory_path.is_dir():
        return f"Processed: 0\nSuccessful: 0\nError: Path is not a directory: {DIRECTORY_PATH}"
    
    # Find all PDF files in the directory
    pdf_files = list(directory_path.glob("*.pdf"))
    pdf_files.extend(directory_path.glob("*.PDF"))  # Handle uppercase extensions
    
    if not pdf_files:
        return f"Processed: 0\nSuccessful: 0\nError: No PDF files found in directory: {DIRECTORY_PATH}"
    
    # Create timestamp-based project name for the book
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"source_{timestamp}"
    
    # Setup directory structure for the entire book project
    dirs = setup_pdf_directory_structure(project_name)
    
    # Process each PDF file (chapter)
    successful_count = 0
    skipped_count = 0
    
    for pdf_file in pdf_files:
        # Create chapter-specific output file name
        chapter_name = pdf_file.stem
        output_txt_path = dirs['raw_ocr'] / f"{chapter_name}.txt"
        
        if output_txt_path.exists() and not OVERWRITE:
            skipped_count += 1
            continue
        
        # Process the PDF using the internal function, but save to the shared project directory
        file_path = pdf_file
        if not file_path.exists():
            continue
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        extracted_text_str = ""
        
        try:
            pages = convert_from_path(str(file_path), dpi=300, fmt="png")
            custom_config = r'--oem 1 --psm 6 -l eng'
            
            processed_pages = 0
            for i, page in enumerate(pages):
                temp_img_path = os.path.join(temp_dir, f"page_{i}.png")
                page.save(temp_img_path)
                
                try:
                    text = pytesseract.image_to_string(temp_img_path, config=custom_config)
                except Exception as e:
                    text = f"[OCR failed on page {i + 1}: {str(e)}]"
                
                extracted_text_str += f"\n\n--- PAGE {i + 1} ---\n\n{text}"
                processed_pages += 1
            
            # Ensure proper UTF-8 encoding
            extracted_text_str = extracted_text_str.encode('utf-8', errors='replace').decode('utf-8')
            
            # Save extracted text to shared raw_ocr directory
            with open(output_txt_path, 'w', encoding='utf-8') as output_file:
                output_file.write(extracted_text_str)
            
            successful_count += 1
            
        except Exception as e:
            pass  # Skip failed files
        
        finally:
            # Clean up temporary files
            for fname in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, fname))
                except Exception:
                    pass
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass
    
    # Create summary message
    total_processed = successful_count + skipped_count
    summary = f"Processed: {total_processed}\nSuccessful: {successful_count}\nOutput: {dirs['raw_ocr']}"
    
    return summary


if __name__ == "__main__":
    mcp.run(transport="stdio")
        