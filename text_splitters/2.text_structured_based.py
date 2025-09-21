from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Object Oriented Programming is a fundamental concept in Python, 
empowering developers to build modular, maintainable and scalable applications.

OOP is a way of organizing code that uses objects and classes to represent real-world 
entities and their behavior. In OOP, object has attributes thing that has specific data 
and can perform certain actions using methods
"""

# Initialize the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20
)

# perform the split
chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))
print(chunks)