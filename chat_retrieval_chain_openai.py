from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import os
from langchain_openai import ChatOpenAI
import settings


os.environ['OPENAI_API_KEY'] = settings.OPENAI_API_KEY
llm = ChatOpenAI()
loader = WebBaseLoader("https://docs.smith.langchain.com")
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()

docs = loader.load()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])
