from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
# from langchain_tavily import TavilySearch

load_dotenv()

class Source(BaseModel):
    '''Schema for the source of the information'''
    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    '''Schema for the agent's response'''
    response: str = Field(description="The response to the user's query")
    sources: List[Source] = Field(default_factory=list, description="The sources of the information")

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    '''
    Tool to search the web for information
    Args:
        query: The query to search for
    Returns:
        The search result
    '''
    print(f"Searching for {query}")
    return tavily.search(query)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":[HumanMessage(content="Find me the best job postings on linkedin for a langchain engineer in Bangalore, Mumbai or Hyderabad")]})
    print(result)

if __name__ == "__main__":
    main()
