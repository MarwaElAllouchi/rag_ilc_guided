from pathlib import Path

from app.core.logging_config import setup_logging
from batch.processing.cycle_parser import CycleParser


setup_logging("INFO")

file_path = Path("data/raw/google_drive_docs/niveaux.docx")

cycles = CycleParser.parse_cycles(file_path)

print("\n=== CYCLES EXTRAITS ===\n")
for cycle in cycles:
    print(cycle)
    print("\n---------------------\n")