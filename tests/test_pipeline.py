from app.core.logging_config import setup_logging
from app.rag.pipeline import RagPipeline


setup_logging("INFO")

pipeline = RagPipeline()

result = pipeline.run("Comment s'inscrire à l'école ?", top_k=5)

print("\n--- RÉPONSE ---\n")
print(result["answer"])

print("\n--- NOMBRE DE DOCUMENTS RÉCUPÉRÉS ---\n")
print(len(result["documents"]))