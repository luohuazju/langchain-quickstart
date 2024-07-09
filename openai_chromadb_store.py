from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import settings
from langchain.text_splitter import TokenTextSplitter

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
vectordb.persist()

