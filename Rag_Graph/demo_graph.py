from dotenv import load_dotenv
import os
from langchain_community.llms import ollama
from typing import Annotated
from typing_extensions TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

llm=ollama(model="llama2")

class State(TypedDict):
  messages:Annotated[list,add_messages]

graph_builder=StateGraph(State)

def chatbot(state:State):
  return {"messages":llm.invoke(state['messages'])}

graph_builder.add_node("chatbot",chatbot)

graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)
graph=graph_builder.compile()

from IPython.display import Image, display
try:
  display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
  pass

while True:
  user_input=input("User: ")
  if user_input.lower() in ["quit","q"]:
    print("Good Bye")
    break
  for event in graph.stream({'messages':("user",user_input)}):
    print(event.values())
    for value in event.values():
      print(value['messages'])
      print("Assistant:",value["messages"].content)