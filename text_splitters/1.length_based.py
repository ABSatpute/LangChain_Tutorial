from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("D:/genAI/text_splitters/Mach4-G-and-M-Code-Reference-Manual.pdf")

docs = loader.load()


splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
    
)

result = splitter.split_documents(docs)
print(result[0].page_content)