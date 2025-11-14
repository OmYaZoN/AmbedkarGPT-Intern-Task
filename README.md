A simple RAG (Retrieval-Augmented Generation) system using:

Ollama (Mistral model)

LangChain

HuggingFace Sentence Embeddings

ChromaDB (vector store)

Loads speech.txt, splits it into chunks, creates embeddings, and answers questions through a CLI.

How to Run

Create and activate virtual environment

Install dependencies:

pip install -r requirements.txt

Install and set up Ollama:

ollama pull mistral

Run the application:

python main.py

Project Files

main.py → Main RAG implementation

requirements.txt → Dependencies

speech.txt → Input document

README.md → Project summary
