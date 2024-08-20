from dotenv import load_dotenv
import os
from langchain_community.llms import ollama

load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

llm=ollama(model="llama2")