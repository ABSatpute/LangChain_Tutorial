from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from requests.exceptions import ChunkedEncodingError
import time

load_dotenv()

# Hugging Face model endpoint with safe parameters
llm = HuggingFaceEndpoint(
    repo_id="google/medgemma-4b-it",  # you can swap to mistralai/Mistral-7B-Instruct-v0.2
    task="text-generation",
    max_new_tokens=512,   # limit output size
    temperature=0.7       # keep answers coherent but creative
)

model = ChatHuggingFace(llm=llm)

# Safe invoke with retry logic
def safe_invoke(model, prompt, retries=3, delay=2):
    for i in range(retries):
        try:
            return model.invoke(prompt)
        except ChunkedEncodingError:
            print(f"⚠️ Response dropped, retrying {i+1}/{retries}...")
            time.sleep(delay)
    raise RuntimeError("❌ Failed after retries due to repeated ChunkedEncodingError")

# Prompts
template1 = PromptTemplate(
    template="Write a detailed report about {topic}. Include recent developments and statistics.",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text:\n{text}",
    input_variables=["text"]
)

# Generate detailed report
prompt1 = template1.invoke({'topic': "Black holes"}).to_string()
result = safe_invoke(model, prompt1)

# Generate summary
prompt2 = template2.invoke({'text': result.content}).to_string()
result1 = safe_invoke(model, prompt2)

print("---- Detailed Report ----")
print(result.content)
print("\n---- 5 Line Summary ----")
print(result1.content)
