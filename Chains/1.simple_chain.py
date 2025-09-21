from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

prompt = PromptTemplate(
    template="generate five interesting facts about {topic}",
    input_variables=["topic"],
)

llm = ChatOpenAI()

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"topic": "artificial intelligence"})

print(result)

chain.get_graph().print_ascii()

