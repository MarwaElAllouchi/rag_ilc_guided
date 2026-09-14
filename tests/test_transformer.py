from pathlib import Path

from app.core.logging_config import setup_logging
from batch.ingestion.sheet_loader import SheetLoader
from batch.processing.transformer import DataTransformer


setup_logging("INFO")

# FAQ
faq_path = Path("data/raw/google_sheets/faq.xlsx")
faq_loader = SheetLoader(faq_path)
faq_df = faq_loader.clean_empty_rows(faq_loader.normalize_columns(faq_loader.load()))
faq_docs = DataTransformer.transform_faq(faq_df)

print("=== FAQ ===")
print(faq_docs[:2])

# Formules / tarifs
formules_path = Path("data/raw/google_sheets/formules_tarifs.xlsx")
formules_loader = SheetLoader(formules_path)
formules_df = formules_loader.clean_empty_rows(
    formules_loader.normalize_columns(formules_loader.load())
)
formules_docs = DataTransformer.transform_formules_tarifs(formules_df)

print("\n=== FORMULES / TARIFS ===")
print(formules_docs[:2])

# Niveaux
niveaux_path = Path("data/raw/google_sheets/niveaux.xlsx")
niveaux_loader = SheetLoader(niveaux_path)
niveaux_df = niveaux_loader.clean_empty_rows(
    niveaux_loader.normalize_columns(niveaux_loader.load())
)
niveaux_docs = DataTransformer.transform_niveaux(niveaux_df)

print("\n=== NIVEAUX ===")
print(niveaux_docs[:2])