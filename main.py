import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,END,add_messages
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_community.tools import TavilySearchResults
from dotenv import find_dotenv,load_dotenv
from langchain_mcp import MCPToolkit
from typing import Annotated,TypedDict


load_dotenv()



GROQAPI_KEY=os.environ["GROQAPI_KEY"]

llm=ChatGroq(
 model="llama3-70b-8192",
 groq_api_key=os.environ.get("GROQAPI_KEY")
)

mcp_toolkit=MCPToolkit.from_server("tavily-mcp")
tools=mcp_toolkit.get_tools()


class AgentState(TypedDict):
    messages:Annotated[list,add_messages]



async def answer_generator(state:AgentState):
    result=await llm.ainvoke(messages=state["messages"],
        tools=tools)
    
    return{
       'messages':state['messages'] + [result]
    }


async def revisor(state:AgentState):
    """Revise the Answer."""
    result=await llm.ainvoke(
           tools=tools,
           messages=[SystemMessage(content="You are a professional answer revisor."
           "Use critique to rewrite or improve the polished version of the answer."
 )]           )
    return {
        'messages':state['messages'] + [result]
    }
      


async def critique(state:AgentState):
    """Critque the answer"""
    result=await llm.ainvoke(
        tools=tools,
        messages=[SystemMessage(content="You are a professional critique agent."
        "Read the message and critique it to find what's missing."
        "Suggest how the answer can be more refined,polished and sourced.")]
    )

    return {
        'messages':state['messages'] + [result]
    }



async def final_answer(state:AgentState):
    """GIVES the final answer"""
    result=await llm.ainvoke(
        tools=tools,
        messages=[SystemMessage(content="You are the final arbiter."
        "Generate the answer."
        "Do not include critique remarks only generate the best polished answer.")]
        )

    return {
        'messages':state['messages'] + [result]
    }




graph=StateGraph(AgentState)


#adding the nodes


graph.add_node("answer_generator",answer_generator)

graph.add_node("revisor",revisor)
graph.add_node("critique",critique)
graph.add_node("final_answer",final_answer)

#adding edges to connect the nodes

graph.add_edge("answer_generator","revisor")
graph.add_edge("revisor","critique")
graph.add_edge("critique","final_answer")

graph.set_entry_point("answer_generator")
workflow=graph.compile()






