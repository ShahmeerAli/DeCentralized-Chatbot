import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,END,add_messages
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_community.tools import TavilySearchResults
from dotenv import find_dotenv,load_dotenv
from typing import Annotated,TypedDict


load_dotenv()



GROQAPI_KEY=os.environ["GROQAPI_KEY"]

llm=ChatGroq(
 model="llama3-70b-8192",
 groq_api_key=os.environ.get("GROQAPI_KEY")
)

search_tools=TavilySearchResults()

tools=[search_tools]

class AgentState(TypedDict):
    messages:Annotated[list,add_messages]



async def answer_generator(state:AgentState):
    result=await llm.ainvoke(tools=tools)
    return{
       'messages':state['messages'] + [result]
    }


async def revisor(state:AgentState):
    """Revise the Answer."""
    result=await llm.ainvoke(
           tools=tools,
           SystemMessage=""
       )
      

async def critique(state:AgentState):
    """Critque the answer"""

async def final_answer(state:AgentState):
    pass

    




