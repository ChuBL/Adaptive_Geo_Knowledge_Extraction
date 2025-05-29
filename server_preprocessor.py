import os
from typing import List
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Preprocessor")

class EntryExtractionResponse(BaseModel):
    """Always use this structure to format your analysis response."""
    identified_entries: List[str] = Field(description="List of identified entry titles/headers from the document")
    document_structure_notes: str = Field(description="Brief notes about the overall document structure and patterns")

@mcp.tool()
def exploratory_data_analysis(FILE_PATH: str) -> str:
    """
    Read and return the content of a text file for AI analysis of entry types and document structure.
    
    The AI agent MUST analyze the returned text content and structure the response using EntryExtractionResponse format:
    - Extract entry titles/headers and put them in a Python list
    - Provide brief structural notes about the document
    - Focus on identifying section headers that could be used for further content extraction
    
    Args:
        FILE_PATH (str): Path to the text file to read.
    
    Returns:
        str: Raw content of the text file for AI analysis.
    """
    
    if not os.path.exists(FILE_PATH):
        return f"Error: File not found: {FILE_PATH}"
    
    if not FILE_PATH.lower().endswith('.txt'):
        return f"Error: File must be a .txt file: {FILE_PATH}"
    
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return content
        
    except Exception as e:
        return f"Error reading file: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")