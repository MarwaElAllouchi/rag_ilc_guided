from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging_config import setup_logging
from batch.processing.document_transformer import DocumentTransformer
from batch.processing.metadata_builder import MetadataBuilder


logger = logging.getLogger(__name__)


class RagBatchRunner:
    """
    Orchestration du pipeline RAG pour les documents non FAQ.

    Ce runner traite principalement :
    - les documents longs (.txt, .docx, .pdf)
    - les sources documentaires générales utiles au RAG

    Il ne traite pas la FAQ hybride, déjà gérée par run_faq_batch.py.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".docx", ".pdf"}
    EXTENSION_PRIORITY = {
        ".txt": 3,
        ".docx": 2,
        ".pdf": 1,
    }

    def __init__(
        self,
        raw_docs_dir: Path,
        rag_output_path: Path,
    ) -> None:
        self.raw_docs_dir = Path(raw_docs_dir)
        self.rag_output_path = Path(rag_output_path)

    def run(self) -> dict[str, Any]:
        """
        Exécute le pipeline RAG documents.
        """
        logger.info("Démarrage du pipeline RAG documents")

        documents = self._process_long_documents()

        self._save_json(documents, self.rag_output_path)

        stats = {
            "documents_count": len(documents),
            "output_file": str(self.rag_output_path),
        }

        logger.info(
            "Pipeline RAG documents terminé : %s documents générés",
            stats["documents_count"],
        )

        return {
            "documents": documents,
            "stats": stats,
        }

    def _process_long_documents(self) -> list[dict[str, Any]]:
        
        """
        Parcourt automatiquement le dossier des documents longs
        et transforme tous les fichiers supportés pour le RAG.

        Règles :
        - ignore les fichiers temporaires Word (~$)
        - préfère TXT > DOCX > PDF si plusieurs fichiers ont le même nom de base
        """

        documents: list[dict[str, Any]] = []

        if not self.raw_docs_dir.exists():
            logger.warning(
                "Dossier documents introuvable : %s",
                self.raw_docs_dir,
            )
            return documents

        candidate_files = self._collect_candidate_files()

        selected_files = self._select_best_files(
            candidate_files
        )

        for file_path in selected_files:

            category = self.detect_category(
                file_path.name
            )

            program = self.detect_program(
                file_path.name
            )

            logger.info(
                "Traitement du document long : %s "
                "(category=%s, program=%s)",
                file_path.name,
                category,
                program,
            )

            doc_chunks = DocumentTransformer.transform_document(
                file_path=file_path,
                source_type="document",
                category=category,
                program=program,
                language="fr",
            )

            enriched_chunks = MetadataBuilder.enrich_documents(
                documents=doc_chunks,
                source_file=str(file_path),
                language="fr",
                default_source_type="document",
                default_category=category,
                default_program=program,
            )

            documents.extend(
                enriched_chunks
            )

        return documents

    def _collect_candidate_files(self) -> list[Path]:
        """
        Collecte les fichiers supportés du dossier brut.
        """
        candidate_files: list[Path] = []

        for file_path in self.raw_docs_dir.iterdir():
            if not file_path.is_file():
                continue

            if file_path.name.startswith("~$"):
                logger.info("Fichier temporaire ignoré : %s", file_path.name)
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                logger.info(
                    "Fichier ignoré (extension non supportée) : %s",
                    file_path.name,
                )
                continue

            candidate_files.append(file_path)

        return candidate_files

    def _select_best_files(self, candidate_files: list[Path]) -> list[Path]:
        """
        Sélectionne le meilleur fichier par nom de base selon la priorité :
        TXT > DOCX > PDF
        """
        selected_files: dict[str, Path] = {}

        for file_path in candidate_files:
            base_name = file_path.stem.lower()
            ext = file_path.suffix.lower()

            if base_name not in selected_files:
                selected_files[base_name] = file_path
                continue

            current_ext = selected_files[base_name].suffix.lower()
            if self.EXTENSION_PRIORITY[ext] > self.EXTENSION_PRIORITY[current_ext]:
                selected_files[base_name] = file_path

        return list(selected_files.values())

    @staticmethod
    def detect_category(file_name: str) -> str:
        """
        Détecte une catégorie simple à partir du nom du fichier.
        """
        name = file_name.lower()

        if "reglement" in name:
            return "reglement"
        if "cours" in name:
            return "cours"
        if "inscription" in name:
            return "inscription"
        if "niveau" in name:
            return "niveau"
        if "contact" in name or "coordonne" in name or "info" in name:
            return "information"

        return "general"
    
        
    @staticmethod
    def detect_program(file_name: str) -> str:
        name = file_name.lower()

        if "anglais" in name or "english" in name:
            return "anglais"

        if "arabe" in name:
            return "arabe"

        return "general"

    @staticmethod
    def _save_json(documents: list[dict[str, Any]], output_path: Path) -> None:
        """
        Sauvegarde les documents transformés dans un fichier JSON.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        logger.info("Fichier JSON sauvegardé : %s", output_path)


def main() -> dict[str, Any]:
    setup_logging("INFO")
    settings = get_settings()

    runner = RagBatchRunner(
        raw_docs_dir=settings.raw_data_path / "google_drive_docs",
        rag_output_path=settings.rag_data_path / "transformed_documents.json",
    )

    return runner.run()


if __name__ == "__main__":
    main()