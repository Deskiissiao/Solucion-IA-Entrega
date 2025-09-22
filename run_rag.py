import argparse
import os
import openai
from dotenv import load_dotenv
from src.retriever import FaissRetriever
from src.prompts import ZERO_SHOT_TEMPLATE

load_dotenv()

def call_llm(prompt, model_name="gpt-4o-mini"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    resp = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        max_tokens=300,
        temperature=0.0
    )
    return resp.choices[0].message.content

def build_context(retrieved):
    parts = []
    for r in retrieved:
        src = r["meta"]["source"]
        mid = r["meta"]["id"]
        parts.append(f"[{src} | {mid}]\n{r['text']}")
    return "\n\n".join(parts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", default="data/faiss_index")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    retriever = FaissRetriever(args.index_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2")

    while True:
        q = input("Pregunta cliente (enter para salir): ").strip()
        if not q:
            break
        retrieved = retriever.retrieve(q, k=args.k)
        context = build_context(retrieved)
        prompt = ZERO_SHOT_TEMPLATE.format(context=context, question=q)
        print("\n--- Contexto usado ---")
        print(context)
        print("\n--- Respuesta LLM ---")
        print(call_llm(prompt))
