from pathlib import Path

from app.core.logging_config import setup_logging
from batch.ingestion.file_loader import FileLoader
from batch.processing.chunker import TextChunker


setup_logging("INFO")

file_path = Path("data/raw/google_drive_docs/reglement_interieur.docx")

loader = FileLoader(file_path)
content = loader.load()

chunker = TextChunker()
chunks = chunker.split_text(content)

print(f"Nombre de chunks : {len(chunks)}")
print("\n--- Premier chunk ---\n")
print(chunks[0])

if len(chunks) > 1:
    print("\n--- Deuxième chunk ---\n")
    print(chunks[1])