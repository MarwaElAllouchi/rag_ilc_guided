from app.core.logging_config import setup_logging
from app.rag.generator import ResponseGenerator
from app.rag.prompt_builder import PromptBuilder


setup_logging("INFO")

documents = [
    {
        "content": "Question : Comment s’inscrire ?\nRéponse : Vous pouvez vous inscrire en ligne ou directement au secrétariat.",
        "metadata": {
            "source_file": "faq.xlsx",
            "category": "inscription",
        },
    },
    {
        "content": "L’inscription est définitive lorsque le dossier est complet.",
        "metadata": {
            "source_file": "reglement_interieur.docx",
            "category": "reglement",
        },
    },
]

messages = PromptBuilder.build_prompt(
    user_question="Comment s'inscrire à l'école ?",
    documents=documents,
)

generator = ResponseGenerator()
answer = generator.generate(messages)

print("\n--- RÉPONSE GÉNÉRÉE ---\n")
print(answer)