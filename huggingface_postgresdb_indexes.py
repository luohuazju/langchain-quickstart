from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
import settings


collection_name = "huggingface_postgresdb_index_python"

CONNECTION_STRING = PGVector.connection_string_from_db_params(
     driver="psycopg2",
     host=settings.POSTGRESDB_HOST,
     port=settings.POSTGRESDB_PORT,
     database=settings.POSTGRESDB_NAME,
     user=settings.POSTGRESDB_USERNAME,
     password=settings.POSTGRESDB_PASSWORD,
)

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=settings.OLLAMA_HOST)

namespace = f"postgresdb/{collection_name}"
record_manager = SQLRecordManager(
    namespace, db_url="postgresql+psycopg2://" +
                      settings.POSTGRESDB_RECORD_USERNAME + ":" +
                      settings.POSTGRESDB_RECORD_PASSWORD + "@" +
                      settings.POSTGRESDB_RECORD_HOST + ":" +
                      settings.POSTGRESDB_RECORD_PORT + "/" +
                      settings.POSTGRESDB_RECORD_NAME
)

store = PGVector(
    collection_name=collection_name,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

# create schema if needed
# record_manager.create_schema()

doc1 = Document(page_content="kitty", metadata={"source": "kitty.txt"})
doc2 = Document(page_content="doggy", metadata={"source": "doggy.txt"})

index(
    [doc1, doc1, doc1, doc1, doc1],
    record_manager,
    store,
    cleanup=None,
    source_id_key="source",
)


def _clear():
    """Hacky helper method to clear content. See the `full` mode section to to understand why it works."""
    index([], record_manager, store, cleanup="full", source_id_key="source")


_clear()

index(
    [doc1, doc2],
    record_manager,
    store,
    cleanup="incremental",
    source_id_key="source",
)

_clear()
all_docs = [doc1, doc2]
index(all_docs, record_manager, store, cleanup="full", source_id_key="source")

docs_with_score = store.similarity_search_with_score("doggy")

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
