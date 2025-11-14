
"""
main.py
Command-line Q&A using local Ollama (mistral), HuggingFace embeddings,
Chroma, and LangChain.
"""

import argparse
from pathlib import Path
import sys


from langchain_text_splitters import CharacterTextSplitter

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains import RetrievalQA



from langchain.schema import Document


def build_or_load_vectorstore(text_path: Path, persist_directory: str = "chroma_db"):
    # take test as input
    txt = text_path.read_text(encoding="utf-8")

    # chunk it into smaller chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=400,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(txt)

    # wrap chunk in document obj
    docs = [Document(page_content=c, metadata={"source": str(text_path)}) for c in chunks]

    # Embeddings 
    hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create ChromaDB
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=hf,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb


def main(args):
    text_path = Path(args.speech_file)
    if not text_path.exists():
        print(f"Error: speech file not found at {text_path}")
        sys.exit(1)

    print("Building/loading vectorstore (this may take a moment)...")
    vectordb = build_or_load_vectorstore(text_path, persist_directory=args.persist_dir)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # using ollama
    print("Instantiating Ollama LLM (model=mistral). Make sure Ollama is installed and model pulled.")
    llm = Ollama(model="mistral", temperature=0)

    # RetrievalQA setup
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # loop
    print("\nReady. Type your question (or 'exit' to quit):")
    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        # retrieval 
        response = qa.invoke({"query": q})
        print("\n>>>", response["result"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech-file", default="speech.txt", help="Path to speech.txt")
    parser.add_argument("--persist-dir", default="chroma_db", help="Chroma persist directory")
    args = parser.parse_args()
    main(args)
