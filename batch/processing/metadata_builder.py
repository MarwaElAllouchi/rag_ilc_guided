from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class MetadataBuilder:
    """
    Standardise et complète les métadonnées des documents RAG.

    Métadonnées principales utilisées par le système :

    - language
    - source_type
    - category
    - program
    - source_file
    - document_id
    - faq_id

    Les valeurs category / program / source_type sont normalisées
    en minuscules afin de garantir la compatibilité avec les filtres
    du Retriever.
    """

    DEFAULT_METADATA = {
        "language": "fr",
        "source_type": "document",
        "category": "general",
        "program": "general",
        "source_file": "unknown",
    }

    # =========================================================
    # UTILITAIRES
    # =========================================================

    @staticmethod
    def _clean_str(
        value: Any,
        default: str = "",
    ) -> str:
        """
        Nettoie une valeur texte simple.
        """

        if value is None:
            return default

        text = str(value).strip()

        if not text:
            return default

        return text

    # =========================================================
    # NORMALISATION DES MÉTADONNÉES
    # =========================================================

    @classmethod
    def _normalize_metadata(
        cls,
        metadata: dict[str, Any],
        source_file: str | None = None,
        language: str = "fr",
        default_source_type: str | None = None,
        default_category: str | None = None,
        default_program: str | None = None,
    ) -> dict[str, Any]:
        """
        Normalise et complète les métadonnées.

        Priorité :

        1. valeur présente dans metadata
        2. valeur default_* fournie par l'appelant
        3. DEFAULT_METADATA
        """

        original_metadata = (
            metadata.copy()
            if isinstance(metadata, dict)
            else {}
        )

        normalized: dict[str, Any] = {}

        # -----------------------------------------------------
        # LANGUE
        # -----------------------------------------------------

        normalized["language"] = cls._clean_str(
            original_metadata.get("language"),
            default=cls._clean_str(
                language,
                default=cls.DEFAULT_METADATA["language"],
            ),
        ).lower()

        # -----------------------------------------------------
        # TYPE DE SOURCE
        # -----------------------------------------------------

        normalized["source_type"] = cls._clean_str(
            original_metadata.get("source_type"),
            default=cls._clean_str(
                default_source_type,
                default=cls.DEFAULT_METADATA["source_type"],
            ),
        ).lower()

        # -----------------------------------------------------
        # CATÉGORIE
        # -----------------------------------------------------

        normalized["category"] = cls._clean_str(
            original_metadata.get("category"),
            default=cls._clean_str(
                default_category,
                default=cls.DEFAULT_METADATA["category"],
            ),
        ).lower()

        # -----------------------------------------------------
        # PROGRAMME
        # -----------------------------------------------------

        normalized["program"] = cls._clean_str(
            original_metadata.get("program"),
            default=cls._clean_str(
                default_program,
                default=cls.DEFAULT_METADATA["program"],
            ),
        ).lower()

        # -----------------------------------------------------
        # FICHIER SOURCE
        #
        # Important :
        # on conserve la casse originale du nom du fichier.
        #
        # MyScol.docx doit rester MyScol.docx
        # -----------------------------------------------------

        if source_file:

            normalized["source_file"] = (
                Path(source_file).name
            )

        else:

            normalized["source_file"] = cls._clean_str(
                original_metadata.get("source_file"),
                default=cls.DEFAULT_METADATA["source_file"],
            )

        # -----------------------------------------------------
        # CONSERVER LES MÉTADONNÉES ADDITIONNELLES
        # -----------------------------------------------------

        reserved_keys = {
            "language",
            "source_type",
            "category",
            "program",
            "source_file",
        }

        for key, value in original_metadata.items():

            if key not in reserved_keys:
                normalized[key] = value

        # -----------------------------------------------------
        # DOCUMENT ID
        # -----------------------------------------------------

        if "document_id" in normalized:

            normalized["document_id"] = (
                cls._clean_str(
                    normalized.get("document_id")
                )
            )

        # -----------------------------------------------------
        # FAQ ID
        # -----------------------------------------------------

        if "faq_id" in normalized:

            normalized["faq_id"] = (
                cls._clean_str(
                    normalized.get("faq_id")
                )
            )

        return normalized

    # =========================================================
    # DOCUMENT ID
    # =========================================================

    @staticmethod
    def _ensure_document_id(
        metadata: dict[str, Any],
        fallback_index: int,
    ) -> dict[str, Any]:
        """
        Ajoute un document_id lorsqu'il est absent.

        Exemple :

        niveaux_arabe.docx
            ↓
        niveaux_arabe_1
        """

        if not metadata.get("document_id"):

            source_file = metadata.get(
                "source_file",
                "unknown",
            )

            stem = (
                Path(source_file).stem
                if source_file
                else "document"
            )

            metadata["document_id"] = (
                f"{stem}_{fallback_index + 1}"
            )

        return metadata

    # =========================================================
    # ENRICHISSEMENT DES DOCUMENTS
    # =========================================================

    @classmethod
    def enrich_documents(
        cls,
        documents: list[dict[str, Any]],
        source_file: str | None = None,
        language: str = "fr",
        default_source_type: str | None = None,
        default_category: str | None = None,
        default_program: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Harmonise les métadonnées des documents.

        - conserve le contenu valide
        - conserve les métadonnées existantes
        - applique les valeurs par défaut
        - normalise category / program / source_type
        - conserve la casse du nom du fichier
        - génère un document_id lorsqu'il est absent
        - ignore les documents invalides
        """

        enriched_docs: list[
            dict[str, Any]
        ] = []

        skipped_count = 0

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
                skipped_count += 1
                continue

            # -------------------------------------------------
            # CONTENU
            # -------------------------------------------------

            content = str(
                document.get(
                    "content",
                    "",
                )
            ).strip()

            if not content:
                skipped_count += 1
                continue

            # -------------------------------------------------
            # MÉTADONNÉES
            # -------------------------------------------------

            metadata = document.get(
                "metadata",
                {},
            )

            if metadata is None:
                metadata = {}

            if not isinstance(
                metadata,
                dict,
            ):
                skipped_count += 1
                continue

            # -------------------------------------------------
            # NORMALISATION
            # -------------------------------------------------

            normalized_metadata = (
                cls._normalize_metadata(
                    metadata=metadata,
                    source_file=source_file,
                    language=language,
                    default_source_type=default_source_type,
                    default_category=default_category,
                    default_program=default_program,
                )
            )

            # -------------------------------------------------
            # DOCUMENT ID
            # -------------------------------------------------

            normalized_metadata = (
                cls._ensure_document_id(
                    metadata=normalized_metadata,
                    fallback_index=index,
                )
            )

            # -------------------------------------------------
            # DOCUMENT FINAL
            # -------------------------------------------------

            enriched_docs.append(
                {
                    "content": content,
                    "metadata": normalized_metadata,
                }
            )

        logger.info(
            "MetadataBuilder : %s documents enrichis, %s ignorés",
            len(enriched_docs),
            skipped_count,
        )

        return enriched_docs