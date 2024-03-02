from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
import settings


os.environ['OPENAI_API_KEY'] = settings.OPENAI_API_KEY

embeddings = OpenAIEmbeddings()
new_db = FAISS.load_local("faiss_index", embeddings)
query = "What did the president say about Ketanji Brown Jackson"
docs = new_db.similarity_search(query)
print(docs[0].page_content)
