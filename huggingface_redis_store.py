from langchain_community.vectorstores.redis import Redis
from langchain_community.embeddings import OllamaEmbeddings
import settings


embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=settings.OLLAMA_HOST)

metadata = [
    {
        "user": "john",
        "age": 18,
        "job": "engineer",
        "credit_score": "high",
    },
    {
        "user": "derrick",
        "age": 45,
        "job": "doctor",
        "credit_score": "low",
    },
    {
        "user": "nancy",
        "age": 94,
        "job": "doctor",
        "credit_score": "high",
    },
    {
        "user": "tyler",
        "age": 100,
        "job": "engineer",
        "credit_score": "high",
    },
    {
        "user": "joe",
        "age": 35,
        "job": "dentist",
        "credit_score": "medium",
    },
]
texts = ["java", "python", "scala", "golang", "rust"]

rds = Redis.from_texts(
    texts,
    embeddings,
    metadatas=metadata,
    redis_url="redis+sentinel://" + settings.REDIS_HOST + ":" + settings.REDIS_PORT + "/" + settings.REDIS_CLUSTER +  "/" + settings.REDIS_DATABASE,
    index_name="users",
    password=settings.REDIS_PASSWORD,
)

print(rds.index_name)

results = rds.similarity_search("java")
print(results[0].page_content)

results = rds.similarity_search("python", k=3)
meta = results[1].metadata
print("Key of the document in Redis: ", meta.pop("id"))
print("Metadata of the document: ", meta)

results = rds.similarity_search_with_relevance_scores("scala", k=5)
for result in results:
    print(f"Content: {result[0].page_content} --- Similiarity: {result[1]}")

# you can also add new documents as follows
new_document = ["php"]
new_metadata = [{"user": "sam", "age": 50, "job": "janitor", "credit_score": "high"}]
# both the document and metadata must be lists
rds.add_texts(new_document, new_metadata)

results = rds.similarity_search("php", k=3)
print(results[0].metadata)

rds.write_schema("redis_schema.yaml")

rds_tool = Redis.from_existing_index(
    redis_url="redis+sentinel://" + settings.REDIS_HOST + ":" + settings.REDIS_PORT + "/" + settings.REDIS_CLUSTER +  "/" + settings.REDIS_DATABASE,
    index_name="users",
    embedding = embeddings,
    schema="redis_schema.yaml",
    password=settings.REDIS_PASSWORD,
)

results = rds_tool.similarity_search("foo", k=3)
meta = results[1].metadata
print("Key of the document in Redis: ", meta.pop("id"))
print("Metadata of the document: ", meta)
