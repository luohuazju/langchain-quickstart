from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import settings
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import (
    StuffDocumentsChain, ConversationalRetrievalChain
)

# chroma persist directory, otherwise it will be in memory
persist_directory = "/Users/carl/work/genai/langchain-quickstart/public"
loader = DirectoryLoader(persist_directory, glob='**/*.txt')
docs = loader.load()
# loader = TextLoader('/Users/carl/work/genai/langchain-quickstart/public/***.txt', encoding='utf8')
# docs = loader.load()

os.environ['OPENAI_API_KEY'] = settings.OPENAI_API_KEY
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
# split the text to make sure it will not go over the limit of token
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
doc_texts = text_splitter.split_documents(docs)
vectordb = Chroma.from_documents(doc_texts, embeddings, persist_directory=persist_directory)
# vectordb.persist()

question = "What did the president say about Ketanji Brown Jackson"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
docs = vectordb.similarity_search(question, k=4)
print(docs)

# vectordb = Chroma.from_documents(docs, embeddings)
retriever = vectordb.as_retriever()
chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    retriever=retriever,
    return_source_documents=True)
chat_history = []
result = chain({"question": question, "chat_history": chat_history})
print(result)
