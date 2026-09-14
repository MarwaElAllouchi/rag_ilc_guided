from pathlib import Path

from app.core.logging_config import setup_logging
from batch.ingestion.sheet_loader import SheetLoader


setup_logging("INFO")

file_path = Path("data/raw/google_sheets/faq.xlsx")

loader = SheetLoader(file_path)

df = loader.load()
df = loader.normalize_columns(df)
df = loader.clean_empty_rows(df)

print(df.head())
print(df.columns.tolist())