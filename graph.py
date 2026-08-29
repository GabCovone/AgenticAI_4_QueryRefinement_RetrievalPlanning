import os
import functools
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from typing import Literal

# Definiamo lo stato prima degli import degli agenti per evitare dipendenze circolari
class GraphState(TypedDict):
    original_query: str
    current_query: str
    retrieved_context: str
    num_refinement: int
    num_planning: int
    next_node: str
    feedback_history: List[str]
    final_answer: str

# --- INIZIALIZZAZIONE CENTRALE DEL MODELLO ---
from huggingface_hub import hf_hub_download
from langchain_community.chat_models import ChatLlamaCpp 

REPO_ID = "bartowski/gemma-2-9b-it-GGUF" 
FILENAME = "gemma-2-9b-it-Q4_K_M.gguf" 
LOCAL_DIR = "./models"
LOCAL_PATH = os.path.join(LOCAL_DIR, FILENAME)

_cached_llm = None

def get_llama_model():
    global _cached_llm
    if _cached_llm is not None:
        return _cached_llm
        
    if not os.path.exists(LOCAL_PATH):
        print(f"[INIT] Download da HuggingFace in corso...")
        os.makedirs(LOCAL_DIR, exist_ok=True)
        hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=LOCAL_DIR)
        
    print("[INIT] Caricamento del modello in graph.py...")
    _cached_llm = ChatLlamaCpp(
        model_path=LOCAL_PATH,
        n_gpu_layers=-1,      
        n_ctx=4096,           
        temperature=0.1,      
        max_tokens=2048,
        repeat_penalty=1.15,
        verbose=False
    )
    return _cached_llm

# Importiamo i nodi dopo aver definito lo stato
from Validator import validator_node
from Refiner import refiner_node
from Planner import planner_node

# --- FUNZIONE DI ROUTING CONDIZIONALE ---
def route_after_validator(state: GraphState) -> Literal["refinement_node", "planning_node", "__end__"]:
    """Legge lo stato aggiornato dal validatore e decide il prossimo nodo."""
    next_action = state.get("next_node")
    if next_action == "route_to_refinement":
        return "refinement_node"
    elif next_action == "route_to_planning":
        return "planning_node"
    else:
        return END

# --- COSTRUZIONE DEL GRAFO ---
def build_graph():
    # 1. Istanziamo il modello una volta sola
    shared_llm = get_llama_model()

    # 2. Inizializziamo il grafo passando la struttura dello Stato
    workflow = StateGraph(GraphState)

    # 3. Aggiungiamo i nodi al grafo (iniettando shared_llm con functools.partial)
    workflow.add_node("validator_node", functools.partial(validator_node, llm=shared_llm))
    workflow.add_node("refinement_node", functools.partial(refiner_node, llm=shared_llm))
    workflow.add_node("planning_node", functools.partial(planner_node, llm=shared_llm))

    # Definiamo il punto di partenza
    workflow.add_edge(START, "validator_node")

    # Aggiungiamo i bordi condizionali in uscita dal validatore
    workflow.add_conditional_edges(
        "validator_node",
        route_after_validator,
        {
            "refinement_node": "refinement_node",
            "planning_node": "planning_node",
            END: END
        }
    )

    # Dopo il refinement, si torna SEMPRE al validatore per controllare il lavoro
    workflow.add_edge("refinement_node", "validator_node")
    
    # Dopo il planning, si torna al validatore per verificare i documenti estratti
    workflow.add_edge("planning_node", "validator_node")

    # Compiliamo il grafo
    app = workflow.compile()
    return app

# --- 4. ESECUZIONE DI TEST ---
if __name__ == "__main__":
    app = build_graph()
    
    # Input iniziale dell'utente
    initial_state = {
        "original_query": "Come funzionano le AI agentiche?",
        "current_query": "Come funzionano le AI agentiche?",
        "retrieved_context": "",
        "num_refinement": 0,
        "num_planning": 0,
        "next_node": "",
        "feedback_history": [],
        "final_answer": ""
    }

    print("Grafo compilato con successo! Avvio esecuzione...")
    
    for output in app.stream(initial_state, {"recursion_limit": 50}):
        for key, value in output.items():
            print(f"\n--- Uscita dal nodo '{key}' ---")