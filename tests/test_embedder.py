from app.core.logging_config import setup_logging
from batch.embeddings.embedder import MistralEmbedder


setup_logging("INFO")

embedder = MistralEmbedder()

texts = [
    "Question : Comment s’inscrire ? Réponse : Vous pouvez vous inscrire en ligne.",
    "Le niveau N2 correspond aux élèves nés en 2018.",
]

embeddings = embedder.embed_texts(texts)

print(f"Nombre d'embeddings : {len(embeddings)}")
print(f"Taille du premier embedding : {len(embeddings[0]) if embeddings else 0}")