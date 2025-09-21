from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The Person's age")
    city: str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Give me the name, age and city of a fictional {place} person. \n {format_instruction} ', 
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
) 

# prompt = template.invoke({'place': 'indian'})

# print(prompt)

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)

chain = template | model |parser
result = chain.invoke({'place': 'indian'})
print(result)