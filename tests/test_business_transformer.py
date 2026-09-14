from pathlib import Path

from app.core.logging_config import setup_logging
from batch.ingestion.sheet_loader import SheetLoader
from batch.processing.business_transformer import BusinessTransformer


setup_logging("INFO")

# Tarifs
tarifs_path = Path("data/raw/google_sheets/formules_tarifs.xlsx")
tarifs_loader = SheetLoader(tarifs_path)
tarifs_df = tarifs_loader.clean_empty_rows(
    tarifs_loader.normalize_columns(tarifs_loader.load())
)

tarifs = BusinessTransformer.transform_formules_tarifs(tarifs_df)
print("\n=== TARIFS ===")
print(tarifs[:3])

# Niveaux
niveaux_path = Path("data/raw/google_sheets/niveaux.xlsx")
niveaux_loader = SheetLoader(niveaux_path)
niveaux_df = niveaux_loader.clean_empty_rows(
    niveaux_loader.normalize_columns(niveaux_loader.load())
)

niveaux_mapping = BusinessTransformer.transform_niveaux_mapping(niveaux_df)
payload = BusinessTransformer.build_cycles_niveaux_payload(niveaux_mapping)

print("\n=== NIVEAUX ===")
print(payload)