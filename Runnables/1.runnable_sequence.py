from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_veriables=["topic"]
)

model = ChatOpenAI()

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the following joke in a concise manner: {joke}",
    input_veriables=["joke"],
    
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
print(chain.invoke({"topic":"AI"}))
