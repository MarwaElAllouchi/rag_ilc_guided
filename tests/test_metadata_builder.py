from pathlib import Path

from app.core.logging_config import setup_logging
from batch.ingestion.sheet_loader import SheetLoader
from batch.processing.transformer import DataTransformer
from batch.processing.metadata_builder import MetadataBuilder


setup_logging("INFO")

faq_path = Path("data/raw/google_sheets/faq.xlsx")

loader = SheetLoader(faq_path)
df = loader.clean_empty_rows(loader.normalize_columns(loader.load()))

documents = DataTransformer.transform_faq(df)
enriched_documents = MetadataBuilder.enrich_documents(
    documents=documents,
    source_file=str(faq_path),
    language="fr",
)

print(enriched_documents[:2])