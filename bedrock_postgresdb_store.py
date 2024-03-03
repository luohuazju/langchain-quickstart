from requests.auth import HTTPBasicAuth
import requests
import base64
import json
import boto3
import settings
from langchain_community.embeddings import BedrockEmbeddings


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

