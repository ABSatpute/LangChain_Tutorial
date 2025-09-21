from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "D:/genAI/document_loaders/books",
    glob = '*.pdf',
    loader_cls = PyPDFLoader
    
)

docs = loader.lazy_load()

# print(docs[0].page_content)
# print(docs[0].metadata)

for doc in docs:
    print(doc.metadata)