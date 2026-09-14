from app.core.logging_config import setup_logging
from app.rag.router import RagRouter


setup_logging("INFO")

router = RagRouter()

result = router.route("Quels sont les tarifs ?", top_k=5)

print("\n--- INTENTION ---\n")
print(result["intent"])

print("\n--- ROUTE ---\n")
print(result["route"])

print("\n--- RÉPONSE ---\n")
print(result["answer"])