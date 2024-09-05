from typing import Annotated
from typing_extensions import TypedDict

from langchain_community.llms import ollama

from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun

Arxiv_W=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
Arxiv_T=ArxivQueryRun(api_wrapper=Arxiv_W)

Wiki_W=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
Wiki_T=WikipediaQueryRun(api_wrapper=Wiki_W)

tools=[Wiki_T]

from langgraph.graph.message import add_messages
class State(TypedDict):
    messages:Annotated[list,add_messages]

from langgraph.graph import StateGraph,START,END

graph_builder=StateGraph(State)

llm=ollama(model="llama2")

llm_with_tools=llm.bind_tools(tools=tools)

def chatbot(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,"chatbot")

from langgraph.prebuilt import ToolNode,tools_condition

tool_node=ToolNode(tools=tools)
graph_builder.add_node("Tools",tool_node)
graph_builder.add_conditional_edges("chatbot",tools_condition)

graph_builder.add_edge("Tools","chatbot")
graph_builder.add_edge("chatbot",END)

graph=graph_builder.compile()