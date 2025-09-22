import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import argparse
from tqdm import tqdm

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

def load_catalog(csv_path: str):
    return pd.read_csv(csv_path, dtype=str).fillna("")

def load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk_tokens))
        i += chunk_size - overlap
    return chunks

def prepare_documents(data_dir: str):
    docs = []
    # catálogo
    cat_path = os.path.join(data_dir, "catalogo.csv")
    if os.path.exists(cat_path):
        df = load_catalog(cat_path)
        for _, r in df.iterrows():
            text = f"{r.get('nombre','')}\n{r.get('descripcion','')}\nPrecio:{r.get('precio','')}\nStock:{r.get('stock','')}\nTallas:{r.get('tallas','')}"
            docs.append({"source": "catalogo", "id": r.get("id"), "text": text})
    # políticas
    pdf_path = os.path.join(data_dir, "politicas.pdf")
    if os.path.exists(pdf_path):
        text = load_pdf_text(pdf_path)
        docs.append({"source": "politicas", "id": "politicas_1", "text": text})
    # FAQs
    faqs_path = os.path.join(data_dir, "faqs.txt")
    if os.path.exists(faqs_path):
        with open(faqs_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                docs.append({"source":"faqs", "id": f"faq_{i}", "text": line.strip()})
    return docs

def build_index(docs, embedding_model_name, index_path):
    model = SentenceTransformer(embedding_model_name)
    texts = []
    metadatas = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({"source": doc["source"], "id": f"{doc['id']}_chunk_{i}"})
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "metadatas.pkl"), "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)
    print(f"Index guardado en {index_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--index_path", default="data/faiss_index")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    docs = prepare_documents(args.data_dir)
    build_index(docs, args.embedding_model, args.index_path)
