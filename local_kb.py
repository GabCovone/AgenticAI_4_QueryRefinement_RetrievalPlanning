import os
import pickle
from datasets import load_dataset
from rank_bm25 import BM25Okapi

INDEX_PATH = "bm25_index.pkl"
CORPUS_PATH = "corpus.pkl"

def build_index(subset_percent=5):
    """
    Builds a local BM25 inverted index using a subset of Wikipedia.
    """
    print(f"\n[LOCAL KB] Inizio scaricamento del {subset_percent}% di Wikipedia (HuggingFace)...")
    try:
        # Load English Wikipedia from 2022
        dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{subset_percent}%]")
    except Exception as e:
        print(f"[LOCAL KB] Errore nel download del dataset: {e}")
        return
        
    corpus = []
    tokenized_corpus = []
    
    print(f"[LOCAL KB] Tokenizzazione del corpus ({len(dataset)} articoli) per BM25 in corso...")
    for doc in dataset:
        # Limiting text to first 2000 chars per article to save RAM and speed up
        text = f"Title: {doc['title']}\n{doc['text'][:2000]}"
        corpus.append(text)
        tokenized_corpus.append(text.lower().split())
        
    print("[LOCAL KB] Costruzione indice invertito BM25...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump(corpus, f)
        
    print(f"[LOCAL KB] Indice costruito con successo! {len(corpus)} documenti salvati.")

def search_index(query: str, top_k=3) -> str:
    """
    Searches the local inverted index for a given query.
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CORPUS_PATH):
        return "Errore: Indice locale non trovato. Esegui build_index() prima di cercare."
    
    # Load index if not already in memory (could be optimized with a global variable, 
    # but loading from pickle is fast enough for 5% subset)
    with open(INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(CORPUS_PATH, "rb") as f:
        corpus = pickle.load(f)
        
    tokenized_query = query.lower().split()
    results = bm25.get_top_n(tokenized_query, corpus, n=top_k)
    
    if not results:
        return "Nessun risultato trovato nella KB locale."
        
    formatted = ""
    for i, res in enumerate(results):
        formatted += f"\n--- Documento {i+1} ---\n{res}...\n"
    return formatted

if __name__ == "__main__":
    # Eseguendo questo script direttamente, si costruisce l'indice (default 5%)
    build_index(5)

