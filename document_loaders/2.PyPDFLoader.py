from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("C:/Users/DELL/Downloads/Mach4-G-and-M-Code-Reference-Manual.pdf")

docs = loader.load()

# print(len(docs))

print(docs[1].page_content)

# print(docs[1].metadata)