from dotenv import load_dotenv
import os

load_dotenv()

import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

from prompts import new_prompt, instruction_str, context
from pdf import dft_engine


tools=[
    QueryEngineTool(
        query_engine=dft_engine,
        metadata=ToolMetadata(
        name="dft_papers", 
        description="This tool is used to query papers that have used Density Functional Theory (DFT)"
        )),
]

llm = OpenAI(model="gpt-3.5-turbo")

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)


while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)

