from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


load_dotenv()  # Load environment variables from .env file

model=ChatOpenAI()

chat_histry = [SystemMessage(content="You are a helpful assistant.")
               ]

while True:
    user_input = input("You:")
    chat_histry.append(HumanMessage(content= user_input) )
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chat. Goodbye!")
        break
    
    result = model.invoke(chat_histry)
    chat_histry.append(AIMessage(content= result.content ) ) 
    print("AI :", result.content)
    
print(chat_histry)