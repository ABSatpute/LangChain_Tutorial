from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


load_dotenv()

model = ChatOpenAI()

schema = [
    ResponseSchema(name="fact_1", description="Fact1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me 3 facts about the topic: {topic} \n {format_instruction} ', 
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.format(topic="Black holes")

# result=model.invoke(prompt)

# final_result=parser.parse(result.content)

# print(final_result)

chain = template | model | parser
result = chain.invoke({'topic': "Black holes"})
print(result)