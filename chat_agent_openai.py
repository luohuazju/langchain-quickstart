from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
import settings


os.environ['OPENAI_API_KEY'] = settings.OPENAI_API_KEY
os.environ['TAVILY_API_KEY'] = settings.TAVILY_API_KEY

llm = ChatOpenAI()
loader = WebBaseLoader("https://docs.smith.langchain.com")
embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
docs = loader.load()
documents = text_splitter.split_documents(docs)

vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()
search = TavilySearchResults()


retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

tools = [retriever_tool, search]

prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor.invoke({"input": "how can langsmith help with testing?"})
# agent_executor.invoke({"input": "what is the weather in SF?"})

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"),
                AIMessage(content="Yes!")]
agent_executor.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})