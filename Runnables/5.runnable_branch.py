from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template="write a detailed report on topic: {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='summarize the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()

parser = StrOutputParser()

reoprt_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())> 300, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
) 

final_chain = RunnableSequence(reoprt_gen_chain, branch_chain)
result=final_chain.invoke({"topic":"China vs America economy"})
print(result)