from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging_config import setup_logging

from batch.embeddings.embedder import MistralEmbedder
from batch.storage.pgvector_store import clear_documents, create_tables, upsert_documents


logger = logging.getLogger(__name__)


class IndexingBatchRunner:
    """
    Orchestration du pipeline d'indexation RAG multi-sources.

    Étapes :
    - charger plusieurs fichiers JSON de documents RAG
    - fusionner les documents
    - valider leur structure
    - dédupliquer les documents
    - générer les embeddings
    - créer les tables si nécessaire
    - indexer dans pgvector
    """

    def __init__(
        self,
        input_files: list[str | Path],
    ) -> None:
        if not input_files:
            raise ValueError(
                "input_files ne peut pas être vide"
            )

        self.input_files = [
            Path(file_path)
            for file_path in input_files
        ]

    # =========================================================
    # PIPELINE PRINCIPAL
    # =========================================================

    def run(self) -> dict[str, Any]:
        """
        Exécute le pipeline complet
        d'indexation multi-sources.
        """

        logger.info(
            "Démarrage du pipeline d'indexation RAG multi-sources"
        )

        raw_documents = (
            self._load_all_documents()
        )

        (
            valid_documents,
            rejected_documents,
        ) = self._validate_documents(
            raw_documents
        )

        deduplicated_documents = (
            self._deduplicate_documents(
                valid_documents
            )
        )

        documents_with_embeddings = (
            self._attach_embeddings(
                deduplicated_documents
            )
        )

        create_tables()

        # Réindexation complète :
        # on supprime les anciens embeddings avant d'insérer
        # la nouvelle version des documents et métadonnées.
        clear_documents()

        upsert_documents(
            documents_with_embeddings
        )

        stats = {
            "input_files_count": len(
                self.input_files
            ),
            "loaded_count": len(
                raw_documents
            ),
            "valid_count": len(
                valid_documents
            ),
            "deduplicated_count": len(
                deduplicated_documents
            ),
            "rejected_count": len(
                rejected_documents
            ),
            "indexed_count": len(
                documents_with_embeddings
            ),
        }

        logger.info(
            (
                "Pipeline d'indexation terminé : "
                "%s fichiers, "
                "%s chargés, "
                "%s valides, "
                "%s après déduplication, "
                "%s rejetés, "
                "%s indexés"
            ),
            stats["input_files_count"],
            stats["loaded_count"],
            stats["valid_count"],
            stats["deduplicated_count"],
            stats["rejected_count"],
            stats["indexed_count"],
        )

        return {
            "indexed_documents":
                documents_with_embeddings,
            "rejected_documents":
                rejected_documents,
            "stats":
                stats,
        }

    # =========================================================
    # CHARGEMENT DES DOCUMENTS
    # =========================================================

    def _load_all_documents(
        self,
    ) -> list[dict[str, Any]]:
        """
        Charge et fusionne tous les documents
        depuis plusieurs fichiers JSON.
        """

        all_documents: list[
            dict[str, Any]
        ] = []

        for file_path in self.input_files:

            documents = (
                self._load_documents_from_file(
                    file_path
                )
            )

            all_documents.extend(
                documents
            )

        logger.info(
            "%s documents chargés au total depuis %s fichiers",
            len(all_documents),
            len(self.input_files),
        )

        return all_documents

    @staticmethod
    def _load_documents_from_file(
        file_path: Path,
    ) -> list[dict[str, Any]]:
        """
        Charge les documents depuis
        un fichier JSON unique.
        """

        if not file_path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {file_path}"
            )

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            documents = json.load(file)

        if not isinstance(
            documents,
            list,
        ):
            raise ValueError(
                f"Le fichier {file_path} "
                "doit contenir une liste de documents."
            )

        logger.info(
            "%s documents chargés depuis %s",
            len(documents),
            file_path,
        )

        return documents

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_documents(
        documents: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """
        Vérifie que chaque document possède :

        - un contenu texte valide
        - un dictionnaire de métadonnées valide

        Les métadonnées métier sont déjà
        préparées en amont par MetadataBuilder.
        """

        valid_documents: list[
            dict[str, Any]
        ] = []

        rejected_documents: list[
            dict[str, Any]
        ] = []

        for index, document in enumerate(
            documents
        ):

            # -------------------------------------------------
            # DOCUMENT INVALIDE
            # -------------------------------------------------

            if not isinstance(
                document,
                dict,
            ):
                rejected_documents.append(
                    {
                        "index": index,
                        "reason": "document_not_dict",
                        "document": document,
                    }
                )
                continue

            # -------------------------------------------------
            # CONTENU
            # -------------------------------------------------

            content = document.get(
                "content",
                "",
            )

            if (
                not isinstance(
                    content,
                    str,
                )
                or not content.strip()
            ):
                rejected_documents.append(
                    {
                        "index": index,
                        "reason": (
                            "missing_or_empty_content"
                        ),
                        "document": document,
                    }
                )
                continue

            # -------------------------------------------------
            # MÉTADONNÉES
            # -------------------------------------------------

            metadata = document.get(
                "metadata",
                {},
            )

            if not isinstance(
                metadata,
                dict,
            ):
                rejected_documents.append(
                    {
                        "index": index,
                        "reason": "invalid_metadata",
                        "document": document,
                    }
                )
                continue

            # -------------------------------------------------
            # DOCUMENT VALIDE
            # -------------------------------------------------

            valid_documents.append(
                {
                    "content": content.strip(),
                    "metadata": metadata,
                }
            )

        logger.info(
            "Validation documents : "
            "%s valides, %s rejetés",
            len(valid_documents),
            len(rejected_documents),
        )

        return (
            valid_documents,
            rejected_documents,
        )

    # =========================================================
    # DÉDUPLICATION
    # =========================================================

    @staticmethod
    def _deduplicate_documents(
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Supprime les doublons à partir de :

        - content
        - source_file
        - category
        - program

        Le champ program est important pour empêcher
        qu'un même contenu appartenant à deux programmes
        différents soit considéré comme un doublon.
        """

        seen: set[
            tuple[
                str,
                str,
                str,
                str,
            ]
        ] = set()

        deduplicated: list[
            dict[str, Any]
        ] = []

        for document in documents:

            content = (
                document
                .get(
                    "content",
                    "",
                )
                .strip()
            )

            metadata = document.get(
                "metadata",
                {},
            )

            source_file = str(
                metadata.get(
                    "source_file",
                    "",
                )
            ).strip()

            category = str(
                metadata.get(
                    "category",
                    "",
                )
            ).strip().lower()

            program = str(
                metadata.get(
                    "program",
                    "",
                )
            ).strip().lower()

            key = (
                content,
                source_file,
                category,
                program,
            )

            if key in seen:
                logger.debug(
                    (
                        "Document dupliqué ignoré : "
                        "source_file=%s, "
                        "category=%s, "
                        "program=%s"
                    ),
                    source_file,
                    category,
                    program,
                )
                continue

            seen.add(
                key
            )

            deduplicated.append(
                document
            )

        logger.info(
            "Déduplication documents : "
            "%s conservés sur %s",
            len(deduplicated),
            len(documents),
        )

        return deduplicated

    # =========================================================
    # EMBEDDINGS
    # =========================================================

    @staticmethod
    def _attach_embeddings(
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Génère les embeddings
        et les attache aux documents.
        """

        if not documents:
            logger.warning(
                "Aucun document valide à vectoriser."
            )
            return []

        embedder = MistralEmbedder()

        texts = [
            document["content"]
            for document
            in documents
        ]

        embeddings = (
            embedder.embed_texts(
                texts
            )
        )

        if (
            len(texts)
            != len(embeddings)
        ):
            raise ValueError(
                (
                    "Nombre d'embeddings incohérent : "
                    f"{len(embeddings)} embeddings "
                    f"pour {len(texts)} textes"
                )
            )

        enriched_documents: list[
            dict[str, Any]
        ] = []

        for (
            document,
            embedding,
        ) in zip(
            documents,
            embeddings,
        ):

            enriched_documents.append(
                {
                    "content":
                        document["content"],
                    "metadata":
                        document["metadata"],
                    "embedding":
                        embedding,
                }
            )

        logger.info(
            "%s documents enrichis avec embeddings",
            len(enriched_documents),
        )

        return enriched_documents


# =============================================================
# MAIN
# =============================================================

def main() -> dict[str, Any]:
    setup_logging(
        "INFO"
    )

    settings = (
        get_settings()
    )

    runner = IndexingBatchRunner(
        input_files=[
            (
                settings.rag_data_path
                / "transformed_documents.json"
            ),
            (
                settings.rag_data_path
                / "faq_documents.json"
            ),
        ],
    )

    return runner.run()


if __name__ == "__main__":
    main()