from requests.auth import HTTPBasicAuth
import requests
import base64
import json
import boto3
import settings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pgvector import PGVector


basic_header = HTTPBasicAuth(settings.CONSUL_USERNAME, settings.CONSUL_PASSWORD)

res = requests.get(settings.CONSUL_HOST + '/v1/kv/bedrock.config', auth = basic_header)
res_encode = res.json()[0]['Value']
res_json = base64.b64decode(res_encode)
config = json.loads(res_json)

default_region = "us-east-1"
AWS_ACCESS_KEY = config.get("aws_key")
AWS_SECRET_KEY = config.get("aws_secret")
AWS_REGION = config.get("aws_region") or default_region

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v1",
)

response = embeddings.embed_query("I am using Bedrock for embeddings as RAG for our GenAI")
print(len(response),type(response))

loader = TextLoader("./state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

CONNECTION_STRING = PGVector.connection_string_from_db_params(
     driver="psycopg2",
     host=settings.POSTGRESDB_HOST,
     port=settings.POSTGRESDB_PORT,
     database=settings.POSTGRESDB_NAME,
     user=settings.POSTGRESDB_USERNAME,
     password=settings.POSTGRESDB_PASSWORD,
)

COLLECTION_NAME = "state_of_the_union_test_titan_bedrock_python"

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
