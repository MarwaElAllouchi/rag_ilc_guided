from app.core.logging_config import setup_logging
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

for msg in messages:
    print(f"\n--- {msg['role'].upper()} ---\n")
    print(msg["content"])