import os
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

POSTGRESDB_HOST = os.getenv('POSTGRESDB_HOST')
POSTGRESDB_PORT = os.getenv('POSTGRESDB_PORT')
POSTGRESDB_NAME = os.getenv('POSTGRESDB_NAME')
POSTGRESDB_USERNAME = os.getenv('POSTGRESDB_USERNAME')
POSTGRESDB_PASSWORD = os.getenv('POSTGRESDB_PASSWORD')

CONSUL_HOST = os.getenv('CONSUL_HOST')
CONSUL_USERNAME = os.getenv('CONSUL_USERNAME')
CONSUL_PASSWORD = os.getenv('CONSUL_PASSWORD')

OLLAMA_HOST = os.getenv('OLLAMA_HOST')
