from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import OllamaEmbeddings

import settings

loader = TextLoader("./state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=settings.OLLAMA_HOST)

CONNECTION_STRING = PGVector.connection_string_from_db_params(
     driver="psycopg2",
     host=settings.POSTGRESDB_HOST,
     port=settings.POSTGRESDB_PORT,
     database=settings.POSTGRESDB_NAME,
     user=settings.POSTGRESDB_USERNAME,
     password=settings.POSTGRESDB_PASSWORD,
)

COLLECTION_NAME = "state_of_the_union_huggingface_python"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)

store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

store.add_documents([Document(page_content="carl likes langchain, python, scala, zeppelin and golang", metadata={"page_id": 1})])
store.add_documents([Document(page_content="carl likes java, springboot, springframework, tomcat, weblogic", metadata={"page_id": 2})])

docs_with_score = db.similarity_search_with_score("carl")

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
