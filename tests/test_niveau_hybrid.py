from app.core.logging_config import setup_logging
from app.rag.router import RagRouter


setup_logging("INFO")

router = RagRouter()

questions = [
    "Mon enfant est né en 2018, quel niveau choisir ?",
    "Explique-moi le niveau N2",
    "Que fait-on au niveau N7 ?",
    "Quel est le contenu du cycle I ?",
]

for question in questions:
    print("\n==============================")
    print("QUESTION :", question)

    result = router.route(question, top_k=5)

    print("INTENTION :", result["intent"])
    print("ROUTE :", result["route"])
    print("RÉPONSE :", result["answer"])