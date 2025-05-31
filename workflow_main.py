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
            # Replace with absolute path to your math_server.py file
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
        }
    }
)


# async main function
async def main():
    tools = await client.get_tools()

    math_tools = [t for t in tools if "multiply" in t.name or "add" in t.name or "divide" in t.name]
    ocr_tools = [t for t in tools if "single_ocr" in t.name or "multi_ocr" in t.name]
    entry_extraction_tools = [t for t in tools if "extract_entries_from_path" in t.name]

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
    
    entry_extraction_agent = create_react_agent(
        model=llm,
        tools=entry_extraction_tools,
        name="entry_extraction_agent",
        prompt="You are a entry extracting agent. When you are provided with a file path, you should call the extract_entries tool to analyze the text content.",
    )


    workflow = create_supervisor(
        [math_agent, ocr_agent, entry_extraction_agent],
        model=llm,
        prompt=(
            "You are a team supervisor managing a math agent and a ocr agent. "
            "For calculations, use math_agent. "
            "For ocr tasks, use ocr_agent."
            "For entry extraction tasks, use entry_extraction_agent. "
        ),
    )
    
    # Compile and run
    app = workflow.compile(name="top_level_supervisor")
    result = await app.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "Please process and extract entries from the dir '/Users/blc/Documents/pyspace/Git_Mindat/mineral-rdf-workflow/data/deposit_seperate/output' and return the result."
            }
        ]
    }, {"recursion_limit": 25})


    for m in result["messages"]:
        m.pretty_print()
    


# execute the main function
if __name__ == "__main__":
    asyncio.run(main())