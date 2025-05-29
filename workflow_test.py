import os
import asyncio
# from mcp.server.fastmcp import FastMCP
# from fastmcp import FastMCP, Client
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
# from langgraph_supervisor.handoff import create_forward_message_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
# from pprint import pprint
# import json
# from langchain_core.messages.base import BaseMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import Runnable
# from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, List, Dict, Any
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.runnable import RunnableLambda
# from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langgraph-mcp-test"

llm = AzureChatOpenAI(
    deployment_name=os.getenv('AZURE_DEPLOYMENT_NAME'), 
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_API_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    temperature=1,
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
    preprocessor_tools = [t for t in tools if "exploratory_data_analysis" in t.name]

    # math_agent = create_react_agent(llm, math_tools)
    # ocr_agent = create_react_agent(llm, ocr_tools)

    math_agent = create_react_agent(
        model=llm,
        tools=math_tools,
        # tools=[multiply],
        name="math_agent",
        prompt="You are a math agent that performs arithmetic calculations. You can use tools of multiply, add, and divide to perform calculations.",
    )
    
    ocr_agent = create_react_agent(
        model=llm,
        tools=ocr_tools,
        name="ocr_agent",
        prompt="You are an OCR agent. When you are provided with a file path or directory, you should call the ocr_tools.",
    )
    
    class EntryExtractionResponse(BaseModel):
        '''Output structure for preprocessor.'''
        identified_entries: List[str] = Field(description="List of identified entry titles/headers from the document")
        document_structure_notes: str = Field(description="Brief notes about the overall document structure and patterns")

    
    preprocessor_agent = create_react_agent(
        model=llm,
        tools=preprocessor_tools,
        name="preprocessor_agent",
        prompt="You are a preprocessor agent. When you are provided with a file path, you should call the preprocessor tools to analyze the text content.",
        response_format=EntryExtractionResponse,
    )

    # forwarding_tool = create_forward_message_tool("supervisor")
    # Create supervisor workflow
    workflow = create_supervisor(
        [math_agent, ocr_agent, preprocessor_agent],
        model=llm,
        prompt=(
            "You are a team supervisor managing a math agent and a ocr agent. "
            "For calculations, use math_agent. "
            "For ocr tasks, use ocr_agent."
            "For preprocessor tasks, use preprocessor_agent. "
        ),
        # output_mode="full_history"
        # tools=[forwarding_tool]
    )
    # Compile and run
    app = workflow.compile()
    result = await app.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "please process the file 'data/deposit_seperate/output/Descriptbe model of carbonatite deposits_ocr_output.txt' and return the content for AI analysis of entry types and document structure."
            }
        ]
    }, {"recursion_limit": 25})
        
    # response = await graph.ainvoke({
    #     "messages": [{"role": "user", "content": "What's 6083 div 869? you must call math agent to calculate"}]
    # })
    # print(result["messages"][-1]["content"])

    # response = await graph.ainvoke({
    #     "messages": [{"role": "user", "content": "the starting page is 6083 divide 869, the ending page is 216 divide 27. please call math agent to calculate and call ocr agent to extract the pages from data/model17-18_deposit_models (dragged).pdf"}]
    # })
    # response = await agent.ainvoke(
    #     {"messages": [{"role": "user", "content": "extract text from data/model17.pdf"}]}
    # )
    
    # response = await agent.ainvoke(
    #     {"messages": [{"role": "user", "content": "what is 33*55"}]}
    # )
    
    for m in result["messages"]:
        m.pretty_print()
    


# execute the main function
if __name__ == "__main__":
    asyncio.run(main())