from langchain_community.vectorstores import Clickhouse, ClickhouseSettings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import settings


loader = TextLoader("./state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=settings.OLLAMA_HOST)

for d in docs:
    d.metadata = {"store_type": "clickhouse"}

settings = ClickhouseSettings(
                database="default",
                host=settings.CLICKHOUSE_HOST,
                port=settings.CLICKHOUSE_PORT,
                username=settings.CLICKHOUSE_USERNAME,
                password=settings.CLICKHOUSE_PASSWORD,
                table="clickhouse_vector_search_lapro_python")

docsearch = Clickhouse.from_documents(docs, embeddings, config=settings)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)

print(docs[0].page_content)

print(str(docsearch))

print(f"Clickhouse Table DDL:\n\n{docsearch.schema}")
