from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

model = ChatOpenAI(model="gpt-4o")


#schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "A list of key themes discussed in the review"]
    summary: Annotated[str, "A brief summary of the review in one sentence"] 
    sentiment: Annotated[str, "The overall sentiment of the review, either positive, negative, or neutral"]
    pros: Annotated[list[str], "A list of positive aspects mentioned in the review"]
    cons: Annotated[list[str], "A list of negative aspects mentioned in the review"]
    
structured_model = model.with_structured_output(Review)
    

result = structured_model.invoke("""I bought this tshirt about 3 weeks ago and I'm extremely satisfied with the quality of this tshirt. It feels really good to wear. The cotton quality is soft, breathable and perfect for indian weather. The fit is neat and works great for casual wears, office as well as outings. Even after some washes, the color and fabric stayed the same which is really impressive. The material doesn't feel cheap, that's what i expect from a premium brand. Overall it's simple yet classic polo tshirt at a fair price. This is definitely worth buying...""")

print(result)
print(result['summary'])
print(result['sentiment'])