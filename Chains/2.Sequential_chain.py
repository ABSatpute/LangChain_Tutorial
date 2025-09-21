from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

prompt1= PromptTemplate(
    template="generate a detailed report on {topic}",
    input_variables=["topic"]
)   

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary of the following text \n {text}',
    input_variables=['text']
)

llm = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | llm | parser  | prompt2 | llm | parser

result = chain.invoke({'topic':'Effect of USA Torriffs on Indian Economy'})
print(result)

chain.get_graph().print_ascii()