from __future__ import annotations

import logging
import time
from typing import Any

from app.core.config import get_settings
from app.rag.generator import (
    GenerationServiceUnavailableError,
    ResponseGenerator,
)
from app.rag.prompt_builder import PromptBuilder
from app.rag.query_analyzer import QueryAnalyzer
from app.rag.retriever import Retriever
from batch.embeddings.embedder import EmbeddingServiceUnavailableError


logger = logging.getLogger(__name__)


class RagPipeline:
    """
    Pipeline RAG de l'Institut ILC.

    Responsabilités :
    - rechercher uniquement dans les documents autorisés par le router
    - appliquer les filtres category / program / source_file
    - vérifier la pertinence des documents
    - utiliser directement une FAQ lorsqu'elle correspond clairement
    - générer une réponse à partir des documents trouvés
    - retourner un fallback sécurisé si l'information n'est pas disponible

    Le pipeline ne décide pas du parcours du chatbot.
    Cette responsabilité appartient à router.py.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = Retriever()
        self.generator = ResponseGenerator()

    # =========================================================
    # UTILITAIRES
    # =========================================================

    @staticmethod
    def _clean_question(
        user_question: str,
    ) -> str:

        if not isinstance(user_question, str):
            return ""

        return user_question.strip()

    @staticmethod
    def _build_response(
        answer: str,
        documents: list[dict[str, Any]] | None = None,
        intent: str = "general",
    ) -> dict[str, Any]:

        return {
            "answer": answer,
            "documents": documents or [],
            "intent": intent,
        }

    @staticmethod
    def _build_no_document_answer(
        intent: str,
    ) -> str:

        return "Je n’ai pas cette information pour le moment."

    # =========================================================
    # FAQ
    # =========================================================

    @staticmethod
    def _extract_faq_answer_from_content(
        content: str,
    ) -> str | None:

        if not isinstance(content, str):
            return None

        marker = "Réponse :"

        if marker not in content:
            return None

        answer = content.split(
            marker,
            1,
        )[1].strip()

        return answer or None

    # =========================================================
    # DISTANCE / PERTINENCE
    # =========================================================

    @staticmethod
    def _get_document_distance(
        document: dict[str, Any],
    ) -> float:
        """
        Retourne la meilleure distance disponible :

        - adjusted_distance
        - distance
        - 999.0 si aucune distance
        """

        adjusted_distance = document.get(
            "adjusted_distance"
        )

        if adjusted_distance is not None:
            return float(adjusted_distance)

        distance = document.get(
            "distance"
        )

        if distance is not None:
            return float(distance)

        return 999.0

    # =========================================================
    # FAQ SHORTCUT
    # =========================================================

    @staticmethod
    def _should_short_circuit_with_faq(
        documents: list[dict[str, Any]],
    ) -> bool:
        """
        Utilise directement la réponse FAQ uniquement lorsqu'elle
        est clairement plus pertinente que les autres documents.
        """

        if not documents:
            return False

        top1 = documents[0]

        metadata = (
            top1.get("metadata", {})
            or {}
        )

        source_type = str(
            metadata.get(
                "source_type",
                "",
            )
        ).lower()

        if source_type != "faq":
            return False

        top1_distance = (
            RagPipeline._get_document_distance(
                top1
            )
        )

        # FAQ pas assez proche.
        if top1_distance > 0.20:
            return False

        # Une seule FAQ très pertinente.
        if len(documents) == 1:
            return True

        top2_distance = (
            RagPipeline._get_document_distance(
                documents[1]
            )
        )

        # La première FAQ doit être nettement meilleure.
        return (
            top2_distance - top1_distance
        ) >= 0.08

    # =========================================================
    # PIPELINE PRINCIPAL
    # =========================================================

    def run(
        self,
        user_question: str,
        top_k: int | None = None,
        intent: str | None = None,
        source_files: list[str] | None = None,
        programs: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:

        pipeline_start_time = time.time()

        cleaned_question = self._clean_question(
            user_question
        )

        # -----------------------------------------------------
        # Question vide
        # -----------------------------------------------------

        if not cleaned_question:
            return self._build_response(
                answer=(
                    "Pouvez-vous préciser votre question, "
                    "s’il vous plaît ?"
                ),
                intent="general",
            )

        # -----------------------------------------------------
        # Intention
        # -----------------------------------------------------

        effective_intent = (
            intent
            or QueryAnalyzer.detect_intent(
                cleaned_question
            )
        )

        # -----------------------------------------------------
        # Catégories par défaut
        #
        # Le router peut toujours fournir explicitement
        # categories=[...].
        # Dans ce cas elles sont prioritaires.
        # -----------------------------------------------------

        categories_by_intent = {
            "niveau": [
                "niveau",
            ],

            "inscription": [
                "inscription",
                "paiement",
            ],

            "tarif": [
                "tarif",
                "paiement",
            ],
        }

        effective_categories = (
            categories
            if categories is not None
            else categories_by_intent.get(
                effective_intent
            )
        )

        logger.info(
            "Pipeline RAG | question='%s' | intent=%s | "
            "categories=%s | source_files=%s | programs=%s",
            cleaned_question,
            effective_intent,
            effective_categories,
            source_files,
            programs,
        )

        # =====================================================
        # RETRIEVAL
        # =====================================================

        retrieval_start_time = time.time()

        try:

            documents = self.retriever.retrieve(
                query=cleaned_question,
                top_k=(
                    top_k
                    if top_k is not None
                    else self.settings.top_k
                ),
                categories=effective_categories,
                source_files=source_files,
                programs=programs,
            )

        except EmbeddingServiceUnavailableError:

            logger.warning(
                "Service d'embedding indisponible."
            )

            logger.info(
                "⏱ Pipeline arrêté pendant retrieval : %.3f sec",
                time.time() - pipeline_start_time,
            )

            return self._build_response(
                answer=(
                    "Le service de recherche est "
                    "momentanément indisponible."
                ),
                intent=effective_intent,
            )

        except Exception:

            logger.exception(
                "Erreur pendant le retrieval"
            )

            logger.info(
                "⏱ Pipeline arrêté pendant retrieval : %.3f sec",
                time.time() - pipeline_start_time,
            )

            return self._build_response(
                answer=(
                    "Une erreur technique est survenue."
                ),
                intent=effective_intent,
            )

        logger.info(
            "⏱ Retrieval : %.3f sec",
            time.time() - retrieval_start_time,
        )

        # =====================================================
        # AUCUN DOCUMENT
        # =====================================================

        if not documents:

            logger.info(
                "Aucun document pertinent trouvé."
            )

            logger.info(
                "⏱ Pipeline total : %.3f sec",
                time.time() - pipeline_start_time,
            )

            return self._build_response(
                answer=self._build_no_document_answer(
                    effective_intent
                ),
                intent=effective_intent,
            )

        # =====================================================
        # CONTRÔLE DE PERTINENCE
        # =====================================================

        best_distance = (
            self._get_document_distance(
                documents[0]
            )
        )

        threshold = (
            self.settings.max_retrieval_distance
        )

        logger.info(
            "Distance=%.3f | threshold=%.3f",
            best_distance,
            threshold,
        )

        if best_distance > threshold:

            logger.warning(
                "Fallback : documents non pertinents."
            )

            logger.info(
                "⏱ Pipeline total : %.3f sec",
                time.time() - pipeline_start_time,
            )

            return self._build_response(
                answer=(
                    "Je n’ai pas cette information "
                    "pour le moment."
                ),
                intent=effective_intent,
            )

        # =====================================================
        # FAQ DIRECTE
        # =====================================================

        if self._should_short_circuit_with_faq(
            documents
        ):

            faq_answer = (
                self._extract_faq_answer_from_content(
                    documents[0].get(
                        "content",
                        "",
                    )
                )
            )

            if faq_answer:

                logger.info(
                    "Réponse directe depuis la FAQ."
                )

                logger.info(
                    "⏱ Pipeline total : %.3f sec",
                    time.time() - pipeline_start_time,
                )

                return self._build_response(
                    answer=faq_answer,
                    documents=[
                        documents[0]
                    ],
                    intent=effective_intent,
                )

        # =====================================================
        # CONTEXTE POUR LE LLM
        # =====================================================

        context_documents = documents[:5]

        # =====================================================
        # CONSTRUCTION DU PROMPT
        # =====================================================

        prompt_start_time = time.time()

        messages = PromptBuilder.build_prompt(
            user_question=cleaned_question,
            documents=context_documents,
        )

        logger.info(
            "⏱ Prompt : %.3f sec",
            time.time() - prompt_start_time,
        )

        # =====================================================
        # GÉNÉRATION
        # =====================================================

        generation_start_time = time.time()

        try:

            answer = self.generator.generate(
                messages
            )

        except GenerationServiceUnavailableError:

            logger.warning(
                "Service de génération indisponible."
            )

            logger.info(
                "⏱ Génération interrompue : %.3f sec",
                time.time() - generation_start_time,
            )

            return self._build_response(
                answer=(
                    "Le service de réponse est "
                    "momentanément indisponible."
                ),
                documents=context_documents,
                intent=effective_intent,
            )

        except Exception:

            logger.exception(
                "Erreur pendant la génération."
            )

            return self._build_response(
                answer=(
                    "Une erreur est survenue "
                    "lors de la génération de la réponse."
                ),
                documents=context_documents,
                intent=effective_intent,
            )

        logger.info(
            "⏱ Génération : %.3f sec",
            time.time() - generation_start_time,
        )

        logger.info(
            "⏱ Pipeline total : %.3f sec",
            time.time() - pipeline_start_time,
        )

        # =====================================================
        # RÉPONSE FINALE
        # =====================================================

        return self._build_response(
            answer=answer,
            documents=context_documents,
            intent=effective_intent,
        )