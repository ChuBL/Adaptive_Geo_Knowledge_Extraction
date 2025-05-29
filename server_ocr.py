import os
import tempfile
from typing import Optional
from pdf2image import convert_from_path
import pytesseract
# from PIL import Image
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("OCR")

@mcp.tool()
def single_ocr(FILE_PATH: str, STARTING_PAGE: Optional[int] = 0, ENDING_PAGE: Optional[int] = None) -> str:
    """
    Extract text from a PDF file using OCR and save results to a text file.
    
    Args:
        FILE_PATH (str): Path to the PDF file.
        STARTING_PAGE (int, optional): The index (0-based) of the first page to process.
        ENDING_PAGE (int, optional): The index (0-based, exclusive) of the last page to process.
    
    Returns:
        str: Success message with file paths and extracted text.
    """
    
    if not os.path.exists(FILE_PATH):
        return f"Error: File not found: {FILE_PATH}"
    
    # Get the directory and filename of the original PDF
    pdf_dir = os.path.dirname(os.path.abspath(FILE_PATH))
    pdf_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
    
    # Create output text file path in the same directory as the PDF
    output_txt_path = os.path.join(pdf_dir, f"{pdf_name}_ocr_output.txt")
    
    # Create a temporary directory to store images
    temp_dir = tempfile.mkdtemp()
    extracted_text_str = ""
    
    try:
        pages = convert_from_path(FILE_PATH, dpi=300, fmt="png")
        custom_config = r'--oem 1 --psm 6 -l eng'
        
        for i, page in enumerate(pages):
            if STARTING_PAGE is not None and i < STARTING_PAGE:
                continue
            if ENDING_PAGE is not None and i >= ENDING_PAGE:
                break
            
            temp_img_path = os.path.join(temp_dir, f"page_{i}.png")
            page.save(temp_img_path)
            
            try:
                text = pytesseract.image_to_string(temp_img_path, config=custom_config)
            except Exception as e:
                text = f"[OCR failed on page {i + 1}: {str(e)}]"
            
            extracted_text_str += f"\n\n--- PAGE {i + 1} ---\n\n{text}"
        
        # Ensure proper UTF-8 encoding
        extracted_text_str = extracted_text_str.encode('utf-8', errors='replace').decode('utf-8')
        
        # Save the extracted text to a file in the same directory as the PDF
        with open(output_txt_path, 'w', encoding='utf-8') as output_file:
            output_file.write(extracted_text_str)
        
        # Return success message with file information
        success_message = f"Successfully performed OCR on file: {os.path.basename(FILE_PATH)}\n"
        success_message += f"Results saved to: {output_txt_path}\n\n"
        success_message += "--- EXTRACTED TEXT ---\n"
        success_message += extracted_text_str
        
        return success_message
        
    except Exception as e:
        return f"Error during OCR processing: {str(e)}"
    
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
def multi_ocr(DIRECTORY_PATH: str, OVERWRITE: Optional[bool] = False) -> str:
    """
    Extract text from all PDF files in a directory using OCR and save results to text files.
    
    Args:
        DIRECTORY_PATH (str): Path to the directory containing PDF files.
        OVERWRITE (bool, optional): Whether to overwrite existing output files. Defaults to False (skip existing files).
    
    Returns:
        str: Summary message with processing results for all PDFs.
    """
    
    if not os.path.exists(DIRECTORY_PATH):
        return f"Error: Directory not found: {DIRECTORY_PATH}"
    
    if not os.path.isdir(DIRECTORY_PATH):
        return f"Error: Path is not a directory: {DIRECTORY_PATH}"
    
    # Create output directory
    output_dir = os.path.join(DIRECTORY_PATH, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files in the directory
    pdf_files = []
    for filename in os.listdir(DIRECTORY_PATH):
        if filename.lower().endswith('.pdf'):
            pdf_files.append(filename)
    
    if not pdf_files:
        return f"No PDF files found in directory: {DIRECTORY_PATH}"
    
    # Process each PDF file
    results = []
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(DIRECTORY_PATH, pdf_filename)
        pdf_name = os.path.splitext(pdf_filename)[0]
        output_txt_path = os.path.join(output_dir, f"{pdf_name}_ocr_output.txt")
        
        # Check if output file already exists and handle according to OVERWRITE flag
        if os.path.exists(output_txt_path) and not OVERWRITE:
            results.append(f"⏭ {pdf_filename} -> Skipped (output file already exists)")
            skipped_count += 1
            continue
        
        # Create a temporary directory to store images for this PDF
        temp_dir = tempfile.mkdtemp()
        extracted_text_str = ""
        
        try:
            pages = convert_from_path(pdf_path, dpi=300, fmt="png")
            custom_config = r'--oem 1 --psm 6 -l eng'
            
            for i, page in enumerate(pages):
                temp_img_path = os.path.join(temp_dir, f"page_{i}.png")
                page.save(temp_img_path)
                
                try:
                    text = pytesseract.image_to_string(temp_img_path, config=custom_config)
                except Exception as e:
                    text = f"[OCR failed on page {i + 1}: {str(e)}]"
                
                extracted_text_str += f"\n\n--- PAGE {i + 1} ---\n\n{text}"
            
            # Ensure proper UTF-8 encoding
            extracted_text_str = extracted_text_str.encode('utf-8', errors='replace').decode('utf-8')
            
            # Save the extracted text to output directory
            with open(output_txt_path, 'w', encoding='utf-8') as output_file:
                output_file.write(extracted_text_str)
            
            overwrite_status = " (overwritten)" if os.path.exists(output_txt_path) and OVERWRITE else ""
            results.append(f"✓ {pdf_filename} -> {pdf_name}_ocr_output.txt{overwrite_status}")
            successful_count += 1
            
        except Exception as e:
            results.append(f"✗ {pdf_filename} -> Error: {str(e)}")
            failed_count += 1
        
        finally:
            # Clean up temporary files for this PDF
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
    summary = f"OCR Processing Complete!\n"
    summary += f"Directory processed: {DIRECTORY_PATH}\n"
    summary += f"Output directory: {output_dir}\n"
    summary += f"Total PDFs found: {len(pdf_files)}\n"
    summary += f"Successfully processed: {successful_count}\n"
    summary += f"Skipped (already exists): {skipped_count}\n"
    summary += f"Failed: {failed_count}\n"
    summary += f"Overwrite mode: {'ON' if OVERWRITE else 'OFF'}\n\n"
    summary += "Processing Results:\n"
    summary += "\n".join(results)
    
    return summary


if __name__ == "__main__":
    mcp.run(transport="stdio")