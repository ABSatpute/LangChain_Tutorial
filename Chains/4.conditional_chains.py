from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

model= ChatOpenAI()

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the feedback should be either positive or negative.")

parser2 = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
    template='classify the following feedback text into positive or negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)



classifier_chain =prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='generate a appropriate response to the Positive feedback \ {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='generate a appropriate response to the Negative feedback \ {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment=='positive', prompt2 | model | parser),
    (lambda x: x.sentiment=='negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "No valid sentiment found.")
)

chain = classifier_chain | branch_chain

print(chain.invoke({"feedback": "The product quality is Low and I am very unsatisfied with my purchase!"}))

chain.get_graph().print_ascii()