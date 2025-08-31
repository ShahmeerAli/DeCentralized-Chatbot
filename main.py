import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_community.tools import TavilySearchResults
from dotenv import find_dotenv,load_dotenv


load_dotenv()


GROQAPI_KEY=os.environ["GROQAPI_KEY"]

llm=ChatGroq(
 model="llama3-70b-8192",
 groq_api_key=os.environ.get("GROQAPI_KEY")
)



