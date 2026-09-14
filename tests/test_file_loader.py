from pathlib import Path

from app.core.logging_config import setup_logging
from batch.ingestion.file_loader import FileLoader


setup_logging("INFO")

file_path = Path("data/raw/google_drive_docs/reglement_interieur.docx")

loader = FileLoader(file_path)
content = loader.load()

print(content[:1000])
print("\n--- Longueur du texte ---")
print(len(content))