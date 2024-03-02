import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import settings


os.environ['OPENAI_API_KEY'] = settings.OPENAI_API_KEY

llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

result = chain.invoke({"input":"how can langserve help with deployment?"})
print(result)
