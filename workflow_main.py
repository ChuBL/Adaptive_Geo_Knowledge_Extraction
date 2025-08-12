import os
import asyncio
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_mcp_adapters.client import MultiServerMCPClient

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langgraph-mcp-test"

llm = AzureChatOpenAI(
    deployment_name=os.getenv('AZURE_DEPLOYMENT_NAME'), 
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_API_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    temperature=0.7,
)

client = MultiServerMCPClient(
    {
        "math": {
            "command": "uv",
            "args": ["run", "server_math.py"],
            "transport": "stdio",
        },
        "ocr": {
            "command": "uv",
            "args": ["run", "server_ocr.py"],
            "transport": "stdio",
        },
        "preprocessor": {
            "command": "uv",
            "args": ["run", "server_preprocessor.py"],
            "transport": "stdio",
        },
        "mindat": {
            "command": "uv",
            "args": ["run", "server_mindat.py"],
            "transport": "stdio",
        },
        "geosciml": {
            "command": "uv",
            "args": ["run", "server_geosciml.py"],
            "transport": "stdio",
        }
    }
)


# async main function
async def main():
    tools = await client.get_tools()
    
    def filter_tools(tools, keywords):
        return [t for t in tools if any(k in t.name for k in keywords)]
    
    math_tools = filter_tools(tools, ["multiply", "add", "divide"])
    ocr_tools = filter_tools(tools, ["single_ocr", "multi_ocr"])
    preprocessing_tools = filter_tools(tools, ["extract_entries_from_path"])
    mindat_tools = filter_tools(tools, ["normalize_mindat_entry"])
    geosciml_tools = filter_tools(tools, ["match_geosciml_vocabularies"])

    math_agent = create_react_agent(
        model=llm,
        tools=math_tools,
        name="math_agent",
        prompt="You are a math agent that performs arithmetic calculations. You can use tools of multiply, add, and divide to perform calculations.",
    )
    
    ocr_agent = create_react_agent(
        model=llm,
        tools=ocr_tools,
        name="ocr_agent",
        prompt="You are an OCR agent. When you are provided with a file path or directory, you should call the ocr_tools.",
    )
    
    preprocessing_agent = create_react_agent(
        model=llm,
        tools=preprocessing_tools,
        name="preprocessing_agent",
        prompt="You are a preprocessing agent only for extracting entries from the input directory. When you are provided with a file path, you should call the extract_entries_from_path tool to analyze the text content. For each calling, you should call the tool only once and return the result to the supervisor agent.",
    )
    
    mindat_agent = create_react_agent(
        model=llm,
        tools=mindat_tools,
        name="mindat_agent",
        prompt="You are a mindat agent. When you are provided with a file directory, you should call the normalize_mindat_entry to process the geological entity information. The tool will return a directory with mindat-normalized JSON files. Once finished, you should return the directory path with mindat-normalized results.",
    )
    
    geosciml_agent = create_react_agent(
        model=llm,
        tools=geosciml_tools,
        name="geosciml_agent",
        prompt="You are a geosciml agent. When you are provided with a file directory, you should call the match_geosciml_vocabularies to match the geosciml vocabulary to the geological files. The tool will return another directory with matched results. Once finished, you should return the directory path with matched results.",
    )


    workflow = create_supervisor(
        [math_agent, ocr_agent, preprocessing_agent, mindat_agent, geosciml_agent],
        model=llm,
        prompt=(
            "You are a team supervisor managing a geological data processing pipeline with the following agents:"
            "AGENTS:"
            "- math_agent: For mathematical calculations"
            "- ocr_agent: For OCR text extraction from images/documents → returns file path with OCR results"
            "- preprocessing_agent: For processing OCR text results into structured data → takes file path/dir, returns output dir with structured entries"
            "- mindat_agent: For mineral/rock entity normalization → takes directory, returns directory with mindat-normalized JSON files  "
            "- geosciml_agent: For GeosciML vocabulary matching → takes directory with normalized data, returns vocabulary-matched results"
            "WORKFLOW:"
            "Standard pipeline: OCR → Entry Extraction → Mindat Normalization → GeosciML Vocabulary Matching"
            "Each agent can also be called independently for single-step processing."
            "USAGE:"
            "- For full pipeline: Start with ocr_agent, then pass results through subsequent agents"
            "- For partial processing: Call any agent directly with appropriate input (file path or directory)"
            "- Each agent (except math_agent) builds upon the previous agent's output but can work standalone"
            "- Do not reject user requests if they provide a file path or directory for processing"
            
        ),
    )
    
    # Compile and run
    app = workflow.compile(name="top_level_supervisor")
    result = await app.ainvoke({
        "messages": [
            {
                "role": "user",
                # "content": "Please help me calculate the result of 123123*88888 using the math agent",
                # "content": "Please help me match the geosciml vocabulary from the directory 'test/data'"
                # "content": "Please help me normalize the processed entries in '...' using the mindat agent. Then, match the geosciml vocabulary using the geosciml agent."
                "content": "Please help me extract the text from the PDF files in 'data/source' using OCR agent, then process the text entries using the preprocessing agent, and normalize the mineral and rock entities using the mindat agent. Finally, match the geosciml vocabulary using the geosciml agent.",
            }
        ]
    }, {"recursion_limit": 25})


    for m in result["messages"]:
        m.pretty_print()
    


# execute the main function
if __name__ == "__main__":
    asyncio.run(main())
    