import os
import glob
from typing import Dict, List, Optional
from langchain_openai import AzureChatOpenAI
import sys, os
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../.."))
sys.path.insert(0, project_root)
import asyncio
import re
# from server_geosciml import parse_ttl_file

def extract_ttl_members(file_path: str) -> Optional[Dict[str, List[str]]]:
    """
    Parse a TTL file and extract member lists.
    First attempts GeoSciML format, then falls back to ICS geological timescale format.
         
    Args:
        file_path (str): Path to the TTL file
             
    Returns:
        Optional[Dict[str, List[str]]]: Dictionary with collection_name as key and member_list as value
                                      Returns None if both formats fail
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except:
        return None
    
    # First attempt: Try GeoSciML format - PRESERVES ORIGINAL DESIGN
    try:
        # Use regex to extract the URI
        collection_pattern = r'<(http://resource\.geosciml\.org/classifier/cgi/[^>]+)>'
        collection_match = re.search(collection_pattern, content, re.MULTILINE | re.DOTALL)
        
        if collection_match:
            base_uri = collection_match.group(1)
            
            # Extract all members using the extracted base URI
            # Pattern to match both single line and multi-line member declarations
            member_pattern = rf'<{re.escape(base_uri)}/([^>]+)>'
            members_matches = re.findall(member_pattern, content, re.MULTILINE)
            
            # Clean, deduplicate, and sort
            members = sorted(list(set(member.strip() for member in members_matches if member.strip())))
            
            if members:
                # Use complete base URI as key, members are the suffix parts
                return {base_uri: members}
    except:
        pass
    
    # Second attempt: Try ICS geological timescale format with improved logic
    try:
        collections = {}
        
        # Find all independent collection definitions (with rdfs:label)
        collection_pattern = r'<(http://resource\.geosciml\.org/classifier/ics/ischart/([^/>]+))>\s*rdfs:label'
        collection_matches = re.findall(collection_pattern, content)
        
        for full_uri, collection_name in collection_matches:
            # Find the position where this URI appears as subject (not as member reference)
            subject_pattern = re.escape(f'<{full_uri}>') + r'\s*rdfs:label'
            subject_match = re.search(subject_pattern, content)
            
            if not subject_match:
                continue
            
            start_pos = subject_match.start()
            
            # Find the start position of next independent collection definition
            # Look for all subsequent collection definitions
            remaining_content = content[start_pos + 1:]
            next_collection_pattern = r'<http://resource\.geosciml\.org/classifier/ics/ischart/[^/>]+>\s*rdfs:label'
            next_match = re.search(next_collection_pattern, remaining_content)
            
            if next_match:
                end_pos = start_pos + 1 + next_match.start()
                block_content = content[start_pos:end_pos]
            else:
                # Handle end of file case
                block_content = content[start_pos:]
            
            # Extract members
            if 'skos:member' in block_content:
                # More precise member extraction: search only in skos:member related lines
                member_lines = []
                lines = block_content.split('\n')
                in_member_section = False
                
                for line in lines:
                    if 'skos:member' in line:
                        in_member_section = True
                        member_lines.append(line)
                    elif in_member_section:
                        # If line contains URI but not rdfs:label or skos:prefLabel, consider it a member line
                        if ('http://resource.geosciml.org/classifier/ics/ischart/' in line and 
                            'rdfs:label' not in line and 
                            'skos:prefLabel' not in line):
                            member_lines.append(line)
                        elif line.strip().endswith(';') or line.strip().endswith('.'):
                            # Reached section end
                            member_lines.append(line)
                            break
                
                # Extract URIs from member lines
                members = []
                for line in member_lines:
                    uri_matches = re.findall(r'http://resource\.geosciml\.org/classifier/ics/ischart/([^/>]+)', line)
                    for member_name in uri_matches:
                        # Exclude collection name itself
                        if member_name != collection_name:
                            members.append(member_name)
                
                # Deduplicate and sort
                unique_members = sorted(list(set(members)))
                
                if unique_members:
                    collections[full_uri] = unique_members
        
        if collections:
            return collections
    
    except Exception as e:
        # Error logging can be added here for debugging
        pass
    
    # Both formats failed
    return None


async def process_ttl_directory(
    directory_path: str,
    output_md_path: str = "ttl_descriptions.md"
) -> Optional[str]:
    """
    Process all TTL files in a directory and generate a markdown file with descriptions.
    
    Args:
        directory_path (str): Path to directory containing TTL files
        output_md_path (str): Path for output markdown file
    """
    
    if os.path.exists(output_md_path):
        print(f"Vocabulary description file {output_md_path} already exists. Skipping generation.")
        return output_md_path
    
    # Initialize LLM
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.3,
    )
    
    # System prompt for LLM
    system_prompt = """You are a geological terminology expert tasked with creating concise descriptions for GeosciML vocabulary files.

Your task: Generate a single sentence description for a TTL vocabulary file based on its URI and member terms.

Requirements:
1. Write in English
2. Use professional, straightforward language
3. Focus on describing the vocabulary's SCOPE/DOMAIN rather than listing specific members
4. Avoid words like "covering" or "including" that might imply other terms are excluded
5. Do not list multiple specific members (reference members will be provided separately)
6. Use URI domain information to understand the vocabulary's conceptual area

Sentence structure examples:
- "This vocabulary defines terms related to [domain/scope]"
- "This vocabulary contains terminology for [conceptual area]"
- "This vocabulary provides standard terms for [field/classification]"

Output: One clear, professional sentence describing what geological concepts this vocabulary addresses."""

    # Find all TTL files
    ttl_files = glob.glob(os.path.join(directory_path, "*.ttl"))
    
    if not ttl_files:
        print(f"No TTL files found in {directory_path}")
        return
    
    # Process each TTL file
    results = []
    
    for ttl_file in ttl_files:
        filename = os.path.basename(ttl_file)
        # print(f"Processing {filename}...")
        
        try:
            # Parse TTL file
            parsed_data = extract_ttl_members(ttl_file)
            
            if not parsed_data:
                print(f"  Warning: Could not parse {filename}")
                continue
            
            # For each collection in the file (usually 1, but ICS might have multiple)
            for uri, members in parsed_data.items():
                if not members:
                    continue
                
                # Get first 5 members for display
                display_members = members[:5]
                
                # Generate description using LLM
                try:
                    description = await generate_description(llm, system_prompt, uri, members)
                    
                    # Store result
                    results.append({
                        'filename': filename,
                        'uri': uri,
                        'members': display_members,
                        'description': description
                    })
                    
                    # print(f"  ✓ Generated description for {filename}")
                    
                except Exception as e:
                    print(f"  Error generating description for {filename}: {e}")
                    results.append({
                        'filename': filename,
                        'uri': uri,
                        'members': display_members,
                        'description': f"Error: Could not generate description"
                    })
                    
                break  # Only process the first collection in each file for simplicity
        
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue
    
    # Generate markdown file
    await write_markdown_file(results, output_md_path)
    # print(f"Generated {output_md_path} with {len(results)} entries")
    return output_md_path


async def generate_description(
    llm: AzureChatOpenAI,
    system_prompt: str,
    uri: str,
    members: List[str]
) -> str:
    """
    Generate description for a single TTL vocabulary using LLM.
    
    Args:
        llm: The LLM instance
        system_prompt: System prompt for the LLM
        uri: Base URI of the vocabulary
        members: List of member terms
        
    Returns:
        str: Generated description
    """
    # Format members for better readability (limit to first 10 for prompt)
    sample_members = members[:10]
    members_text = ", ".join([f"'{member}'" for member in sample_members])
    
    if len(members) > 10:
        members_text += f" ... (and {len(members) - 10} more)"
    
    human_prompt = f"""URI: {uri}
Members: [{members_text}]

Generate description:"""
    
    # Try up to 3 times
    for attempt in range(3):
        try:
            response = await llm.ainvoke([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            description = str(response.content).strip()
            
            # Clean up the response (remove quotes if present)
            if description.startswith('"') and description.endswith('"'):
                description = description[1:-1]
            
            return description
            
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise e
            continue
    
    raise RuntimeError("Failed to generate description after 3 attempts")


async def write_markdown_file(results: List[Dict], output_path: str) -> None:
    """
    Write results to markdown file.
    
    Args:
        results: List of dictionaries containing filename, members, and description
        output_path: Path to output markdown file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # f.write("# GeosciML Vocabulary Descriptions\n\n")
        # f.write("This file contains descriptions of GeosciML vocabulary files and their sample members.\n\n")
        
        for result in results:
            filename = result['filename']
            members = result['members']
            description = result['description']
            
            # Write filename
            f.write(f"## {filename}\n\n")
            
            # # Write first 5 members
            # f.write("**Sample Members:**\n")
            # for member in members:
            #     f.write(f"- {member}\n")
            # f.write("\n")
            
            # Write description
            f.write(f"**Description:** {description}\n\n")
            f.write("---\n\n")
            
            




async def generate_vocab_descriptions(VOCAB_PATH: str = "./data/vocabularies",
                                       OUTPUT_PATH: str = "./data/vocabularies/geosciml_descriptions.md"
                                      ) -> Optional[str]:
    """
    The main function to run the example usage of processing TTL files.
    """
    # check if output file already exists
    if os.path.exists(OUTPUT_PATH):
        print(f"Skipping vocabulary descriptions generation - file already exists: {OUTPUT_PATH}")
        return OUTPUT_PATH
    
    # Process all TTL files in a directory
    output_path = await process_ttl_directory(
        directory_path=VOCAB_PATH,
        output_md_path=OUTPUT_PATH
    )
    return output_path
        
if __name__ == "__main__":

    # Run the example usage
    # asyncio.run(generate_vocab_descriptions())
    
    print(extract_ttl_members("test/geosci_test/test_ics.ttl"))