import os
from pathlib import Path
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
DATA_DIR = Path("data")  # put .txt files here
INDEX_PATH = "faiss_index"
def load_and_chunk(paths: List[Path]):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    for p in paths:
        loader = TextLoader(str(p), encoding="utf-8")
        loaded = loader.load()
        # add metadata filename
        for d in loaded:
            d.metadata["source"] = p.name
        docs.extend(splitter.split_documents(loaded))
    return docs
def build_or_load_index(docs):
    embeddings = OpenAIEmbeddings()
    if Path(INDEX_PATH).exists():
        db = FAISS.load_local(INDEX_PATH, embeddings)
    else:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(INDEX_PATH)
    return db
def create_chain(db):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  # pick suitable model
    retriever = db.as_retriever(search_kwargs={"k": 6})
    chain = ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=True)
    return chain

def main():
    
    txts = list(DATA_DIR.glob("*.txt"))
    if not txts:
        print("Place some .txt files in ./data and run again.")
        return
    docs = load_and_chunk(txts)
    db = build_or_load_index(docs)
    chain = create_chain(db)

    chat_history = []
    print("Personal RAG chatbot ready. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ")
        if q.lower() in ("exit", "quit"):
            break
        res = chain({"question": q, "chat_history": chat_history})
        answer = res["answer"]
        print("\nBot:", answer)
        sources = {d.metadata.get("source") for d in res.get("source_documents", [])}
        print("Sources:", ", ".join(sources))
        chat_history.append((q, answer))

if __name__ == "__main__":
    main()
