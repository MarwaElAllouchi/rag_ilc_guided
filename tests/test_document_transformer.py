from pathlib import Path

from app.core.logging_config import setup_logging
from batch.processing.document_transformer import DocumentTransformer


setup_logging("INFO")

file_path = Path("data/raw/google_drive_docs/reglement_interieur.docx")

documents = DocumentTransformer.transform_document(
    file_path=file_path,
    source_type="document",
    category="reglement",
    language="fr",
)

print(f"Nombre de chunks/documents : {len(documents)}")
print("\n--- Premier document ---\n")
print(documents[0])

if len(documents) > 1:
    print("\n--- Deuxième document ---\n")
    print(documents[1])