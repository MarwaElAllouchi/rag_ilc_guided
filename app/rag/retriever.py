from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from sqlalchemy import or_, select

from app.core.config import get_settings
from batch.embeddings.embedder import (
    EmbeddingServiceUnavailableError,
    MistralEmbedder,
)
from batch.storage.database import SessionLocal
from batch.storage.pgvector_store import DocumentEmbedding


logger = logging.getLogger(__name__)


class Retriever:
    """
    Retriever RAG de l'Institut ILC.

    Responsabilités :
    - recherche vectorielle via pgvector
    - filtrage strict par catégorie
    - filtrage strict par programme
    - filtrage strict par fichier source
    - reranking lexical léger
    - priorité légère aux FAQ pertinentes

    Le Retriever ne décide pas du parcours utilisateur.
    Les filtres métier sont fournis par router.py / pipeline.py.
    """

    FAQ_PRIORITY_BONUS = 0.03
    EXACT_QUESTION_BONUS = 0.08
    HIGH_OVERLAP_BONUS = 0.06
    MEDIUM_OVERLAP_BONUS = 0.04
    LOW_OVERLAP_BONUS = 0.02

    CATEGORY_MATCH_BONUS = 0.02

    FAQ_SOURCE_TYPE = "faq"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = MistralEmbedder()

    # =========================================================
    # NETTOYAGE
    # =========================================================

    @staticmethod
    def _clean_query(
        query: str,
    ) -> str:

        if not isinstance(query, str):
            return ""

        return query.strip()

    @staticmethod
    def _normalize_values(
        values: list[str] | None,
    ) -> list[str]:
        """
        Normalisation pour les métadonnées standardisées :

        - category
        - source_type
        - program

        Ces valeurs sont stockées en minuscules.
        """

        if not values:
            return []

        cleaned: list[str] = []

        for value in values:

            if not isinstance(value, str):
                continue

            normalized = value.strip().lower()

            if (
                normalized
                and normalized not in cleaned
            ):
                cleaned.append(normalized)

        return cleaned

    @staticmethod
    def _normalize_source_files(
        values: list[str] | None,
    ) -> list[str]:
        """
        Les noms de fichiers conservent leur casse originale.

        Exemple :
        MyScol.docx

        ne doit pas devenir :
        myscol.docx
        """

        if not values:
            return []

        cleaned: list[str] = []

        for value in values:

            if not isinstance(value, str):
                continue

            normalized = value.strip()

            if (
                normalized
                and normalized not in cleaned
            ):
                cleaned.append(normalized)

        return cleaned

    # =========================================================
    # NORMALISATION TEXTE
    # =========================================================

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:

        if not isinstance(value, str):
            return ""

        text = value.lower().strip()

        text = unicodedata.normalize(
            "NFD",
            text,
        )

        text = "".join(
            char
            for char in text
            if unicodedata.category(char) != "Mn"
        )

        text = re.sub(
            r"[^\w\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text

    @classmethod
    def _tokenize(
        cls,
        value: str,
    ) -> list[str]:

        text = cls._normalize_text(value)

        if not text:
            return []

        return text.split()

    # =========================================================
    # BONUS QUESTION / CONTENU
    # =========================================================

    @classmethod
    def _question_overlap_bonus(
        cls,
        query: str,
        content: str,
    ) -> float:

        normalized_query = cls._normalize_text(
            query
        )

        normalized_content = cls._normalize_text(
            content
        )

        if (
            not normalized_query
            or not normalized_content
        ):
            return 0.0

        if normalized_query in normalized_content:
            return cls.EXACT_QUESTION_BONUS

        query_words = set(
            cls._tokenize(normalized_query)
        )

        content_words = set(
            cls._tokenize(normalized_content)
        )

        if (
            not query_words
            or not content_words
        ):
            return 0.0

        overlap_ratio = (
            len(query_words & content_words)
            / len(query_words)
        )

        if overlap_ratio >= 0.8:
            return cls.HIGH_OVERLAP_BONUS

        if overlap_ratio >= 0.6:
            return cls.MEDIUM_OVERLAP_BONUS

        if overlap_ratio >= 0.4:
            return cls.LOW_OVERLAP_BONUS

        return 0.0

    # =========================================================
    # BONUS LEXICAL
    # =========================================================

    @classmethod
    def _keyword_overlap_bonus(
        cls,
        query: str,
        content: str,
    ) -> float:

        query_words = [
            word
            for word in cls._tokenize(query)
            if len(word) >= 4
        ]

        normalized_content = (
            cls._normalize_text(content)
        )

        if (
            not query_words
            or not normalized_content
        ):
            return 0.0

        overlap_count = sum(
            1
            for word in query_words
            if word in normalized_content
        )

        if overlap_count >= 5:
            return 0.08

        if overlap_count >= 4:
            return 0.06

        if overlap_count >= 3:
            return 0.04

        if overlap_count >= 2:
            return 0.02

        return 0.0

    # =========================================================
    # CATÉGORIES
    # =========================================================

    @classmethod
    def _infer_query_categories(
        cls,
        query: str,
    ) -> set[str]:
        """
        Inférence légère utilisée uniquement pour le reranking.

        Elle ne remplace jamais les filtres stricts envoyés
        par le pipeline.
        """

        q = cls._normalize_text(query)

        inferred: set[str] = set()

        if any(
            word in q
            for word in (
                "tarif",
                "prix",
                "cout",
                "payer",
                "paiement",
            )
        ):
            inferred.add("tarif")
            inferred.add("paiement")

        if any(
            word in q
            for word in (
                "inscription",
                "inscrire",
                "dossier",
                "documents",
                "admission",
            )
        ):
            inferred.add("inscription")

        if any(
            word in q
            for word in (
                "niveau",
                "cycle",
                "classe",
                "test de niveau",
                "annee",
            )
        ):
            inferred.add("niveau")

        return inferred

    @classmethod
    def _category_bonus(
        cls,
        query: str,
        metadata: dict[str, Any],
    ) -> float:

        doc_category = str(
            metadata.get(
                "category",
                "",
            )
        ).strip().lower()

        if not doc_category:
            return 0.0

        inferred_categories = (
            cls._infer_query_categories(query)
        )

        if doc_category in inferred_categories:
            return cls.CATEGORY_MATCH_BONUS

        return 0.0

    # =========================================================
    # RERANKING
    # =========================================================

    def _apply_generic_reranking(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        reranked_documents: list[
            dict[str, Any]
        ] = []

        for document in documents:

            metadata = (
                document.get(
                    "metadata",
                    {},
                )
                or {}
            )

            source_type = str(
                metadata.get(
                    "source_type",
                    "",
                )
            ).strip().lower()

            distance = float(
                document.get(
                    "distance",
                    999.0,
                )
            )

            content = str(
                document.get(
                    "content",
                    "",
                )
            )

            bonus = 0.0

            if source_type == self.FAQ_SOURCE_TYPE:
                bonus += self.FAQ_PRIORITY_BONUS

            bonus += self._question_overlap_bonus(
                query=query,
                content=content,
            )

            bonus += self._keyword_overlap_bonus(
                query=query,
                content=content,
            )

            bonus += self._category_bonus(
                query=query,
                metadata=metadata,
            )

            adjusted_distance = max(
                0.0,
                distance - bonus,
            )

            enriched_document = dict(document)

            enriched_document[
                "rerank_bonus"
            ] = bonus

            enriched_document[
                "adjusted_distance"
            ] = adjusted_distance

            reranked_documents.append(
                enriched_document
            )

        reranked_documents.sort(
            key=lambda document: (
                document.get(
                    "adjusted_distance",
                    999.0,
                ),
                document.get(
                    "distance",
                    999.0,
                ),
            )
        )

        return reranked_documents

    # =========================================================
    # RETRIEVAL
    # =========================================================

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        categories: list[str] | None = None,
        source_types: list[str] | None = None,
        source_files: list[str] | None = None,
        programs: list[str] | None = None,
        max_distance: float | None = None,
    ) -> list[dict[str, Any]]:

        cleaned_query = self._clean_query(
            query
        )

        if not cleaned_query:

            logger.warning(
                "Requête vide reçue dans Retriever."
            )

            return []

        k = (
            top_k
            if top_k is not None
            else self.settings.top_k
        )

        if k <= 0:
            raise ValueError(
                "top_k doit être strictement positif"
            )

        # -----------------------------------------------------
        # Normalisation des filtres
        # -----------------------------------------------------

        normalized_categories = (
            self._normalize_values(
                categories
            )
        )

        normalized_source_types = (
            self._normalize_values(
                source_types
            )
        )

        normalized_source_files = (
            self._normalize_source_files(
                source_files
            )
        )

        normalized_programs = (
            self._normalize_values(
                programs
            )
        )

        logger.info(
            "Recherche sémantique lancée "
            "(top_k=%s, categories=%s, "
            "source_types=%s, source_files=%s, "
            "programs=%s, max_distance=%s)",
            k,
            normalized_categories or None,
            normalized_source_types or None,
            normalized_source_files or None,
            normalized_programs or None,
            max_distance,
        )

        # =====================================================
        # EMBEDDING DE LA QUESTION
        # =====================================================

        try:

            query_embedding = (
                self.embedder.embed_single_text(
                    cleaned_query
                )
            )

        except EmbeddingServiceUnavailableError:

            logger.warning(
                "Service d'embedding indisponible "
                "pour la requête."
            )

            raise

        except Exception:

            logger.exception(
                "Erreur lors de la vectorisation "
                "de la requête utilisateur."
            )

            raise

        # On récupère davantage de candidats pour permettre
        # au reranker de travailler correctement.
        initial_k = max(
            k * 2,
            8,
        )

        # =====================================================
        # REQUÊTE POSTGRESQL / PGVECTOR
        # =====================================================

        with SessionLocal() as session:

            distance_expr = (
                DocumentEmbedding
                .embedding
                .cosine_distance(
                    query_embedding
                )
            )

            stmt = select(
                DocumentEmbedding,
                distance_expr.label(
                    "distance"
                ),
            )

            # -------------------------------------------------
            # CATÉGORIE
            # -------------------------------------------------

            if normalized_categories:

                stmt = stmt.where(
                    or_(
                        *[
                            (
                                DocumentEmbedding
                                .metadata_json[
                                    "category"
                                ]
                                .as_string()
                                == category
                            )
                            for category
                            in normalized_categories
                        ]
                    )
                )

                logger.info(
                    "Filtre catégories : %s",
                    normalized_categories,
                )

            # -------------------------------------------------
            # TYPE DE SOURCE
            # -------------------------------------------------

            if normalized_source_types:

                stmt = stmt.where(
                    or_(
                        *[
                            (
                                DocumentEmbedding
                                .metadata_json[
                                    "source_type"
                                ]
                                .as_string()
                                == source_type
                            )
                            for source_type
                            in normalized_source_types
                        ]
                    )
                )

                logger.info(
                    "Filtre types de source : %s",
                    normalized_source_types,
                )

            # -------------------------------------------------
            # FICHIER SOURCE
            # -------------------------------------------------

            if normalized_source_files:

                stmt = stmt.where(
                    or_(
                        *[
                            (
                                DocumentEmbedding
                                .metadata_json[
                                    "source_file"
                                ]
                                .as_string()
                                == source_file
                            )
                            for source_file
                            in normalized_source_files
                        ]
                    )
                )

                logger.info(
                    "Filtre fichiers sources : %s",
                    normalized_source_files,
                )

            # -------------------------------------------------
            # PROGRAMME
            # -------------------------------------------------

            if normalized_programs:

                stmt = stmt.where(
                    or_(
                        *[
                            (
                                DocumentEmbedding
                                .metadata_json[
                                    "program"
                                ]
                                .as_string()
                                == program
                            )
                            for program
                            in normalized_programs
                        ]
                    )
                )

                logger.info(
                    "Filtre programmes : %s",
                    normalized_programs,
                )

            # -------------------------------------------------
            # DISTANCE MAXIMALE OPTIONNELLE
            # -------------------------------------------------

            if max_distance is not None:

                stmt = stmt.where(
                    distance_expr <= max_distance
                )

            stmt = (
                stmt
                .order_by(distance_expr)
                .limit(initial_k)
            )

            results = session.execute(
                stmt
            ).all()

            documents = [
                {
                    "id": (
                        row.DocumentEmbedding.id
                    ),
                    "doc_key": (
                        row.DocumentEmbedding.doc_key
                    ),
                    "content": (
                        row.DocumentEmbedding.content
                    ),
                    "metadata": (
                        row.DocumentEmbedding.metadata_json
                    ),
                    "distance": float(
                        row.distance
                    ),
                }
                for row in results
            ]

        # =====================================================
        # RERANKING
        # =====================================================

        documents = (
            self._apply_generic_reranking(
                query=cleaned_query,
                documents=documents,
            )
        )

        documents = documents[:k]

        logger.info(
            "Retriever : %s documents récupérés.",
            len(documents),
        )

        return documents