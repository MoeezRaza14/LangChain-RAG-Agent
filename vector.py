from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
df = pd.read_csv("C:\\F\\ms_vs_code_files\\Dataset\\realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
db_location = "./chroma_langchain_db" 
add_documents = not os.path.exists(db_location)
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)
if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content=str(row.get("Title","")) + " " + str(row.get("Review","")),
            metadata={"rating": row.get("Rating"), "date": row.get("Date")},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
    vector_store.add_documents(documents=documents, ids=ids)
    try:
        vector_store.persist()
    except Exception:
        pass
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
