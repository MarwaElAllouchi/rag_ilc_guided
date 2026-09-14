from app.core.logging_config import setup_logging
from app.rag.retriever import Retriever

setup_logging("INFO")

print("1. Initialisation du retriever...")
retriever = Retriever()

query = "Puis-je payer en plusieurs fois ?"
print("2. Lancement de la recherche...")
results = retriever.retrieve(query, top_k=5)

print("3. Recherche terminée.")
print(f"Nombre de résultats : {len(results)}")
print("\n--- Premier résultat ---\n")
print(results[0] if results else "Aucun résultat")