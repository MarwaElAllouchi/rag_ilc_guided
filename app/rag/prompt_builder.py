from __future__ import annotations

import logging
import re
from typing import Any

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Construit le prompt envoyé au LLM à partir :

    - de la question utilisateur
    - des documents récupérés par le Retriever

    Le PromptBuilder ne décide pas du parcours du chatbot.
    Il utilise uniquement les extraits déjà sélectionnés
    et filtrés par le Router / Pipeline / Retriever.
    """

    # =========================================================
    # NETTOYAGE TEXTE
    # =========================================================

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:
        """
        Nettoie légèrement un texte avant de l'inclure
        dans le prompt.
        """

        if value is None:
            return ""

        text = str(value).strip()

        if not text:
            return ""

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text

    # =========================================================
    # CONSTRUCTION DU CONTEXTE
    # =========================================================

    @classmethod
    def build_context(
        cls,
        documents: list[dict[str, Any]],
        max_documents: int | None = None,
    ) -> str:
        """
        Construit le contexte transmis au LLM à partir
        des documents déjà sélectionnés par le Retriever.
        """

        if not documents:
            return ""

        settings = get_settings()

        effective_max_documents = (
            max_documents
            if max_documents is not None
            else settings.max_context_chunks
        )

        selected_documents = documents[
            :effective_max_documents
        ]

        context_parts: list[str] = []

        for index, document in enumerate(
            selected_documents,
            start=1,
        ):
            content = cls._clean_text(
                document.get(
                    "content",
                    "",
                )
            )

            if not content:
                continue

            metadata = (
                document.get(
                    "metadata",
                    {},
                )
                or {}
            )

            source_file = cls._clean_text(
                metadata.get(
                    "source_file",
                    "inconnu",
                )
            )

            category = cls._clean_text(
                metadata.get(
                    "category",
                    "general",
                )
            )

            program = cls._clean_text(
                metadata.get(
                    "program",
                    "general",
                )
            )

            source_type = cls._clean_text(
                metadata.get(
                    "source_type",
                    "document",
                )
            )

            context_parts.append(
                "\n".join(
                    [
                        f"[Extrait {index}]",
                        f"Programme: {program}",
                        f"Catégorie: {category}",
                        f"Type: {source_type}",
                        f"Fichier: {source_file}",
                        f"Texte: {content}",
                    ]
                )
            )

        context = "\n\n".join(
            context_parts
        )

        logger.info(
            "Contexte construit : %s extraits inclus",
            len(context_parts),
        )

        return context

    # =========================================================
    # SYSTEM PROMPT
    # =========================================================

    @staticmethod
    def build_system_prompt() -> str:
        """
        Instructions générales données au modèle.

        Le modèle doit uniquement reformuler les informations
        présentes dans les extraits fournis.
        """

        return (
            "Vous êtes l'assistant virtuel de "
            "l'Institut Langues et Cultures (ILC).\n"
            "\n"

            "Vous répondez aux parents, élèves et visiteurs "
            "de manière claire, polie, naturelle et concise.\n"
            "\n"

            "RÈGLES OBLIGATOIRES :\n"
            "- Répondez uniquement à partir des extraits fournis.\n"
            "- N'utilisez jamais vos connaissances générales.\n"
            "- N'inventez aucune information.\n"
            "- Ne complétez jamais une information absente.\n"
            "- Ne transformez jamais une supposition en fait.\n"
            "- Ne déduisez jamais une règle qui n'est pas explicitement indiquée.\n"
            "\n"

            "PROGRAMMES :\n"
            "- Respectez strictement le programme indiqué dans les extraits.\n"
            "- Ne mélangez jamais les informations du programme arabe "
            "avec celles du programme anglais.\n"
            "- Un niveau arabe comme CP1, CP2, N1 ou un Cycle "
            "ne doit jamais être présenté comme un niveau d'anglais.\n"
            "- Un Level anglais ne doit jamais être présenté "
            "comme un niveau du programme arabe.\n"
            "\n"

            "UTILISATION DES EXTRAITS :\n"
            "- Reformulez les informations utiles pour répondre directement "
            "à la question.\n"
            "- Si plusieurs extraits donnent des informations complémentaires "
            "sur le même sujet, vous pouvez les combiner.\n"
            "- N'ajoutez aucune information qui n'apparaît pas dans les extraits.\n"
            "- Si les extraits donnent seulement une partie de la réponse, "
            "répondez uniquement avec cette partie.\n"
            "- Si les extraits ne permettent pas de confirmer une information, "
            "dites clairement qu'elle n'est pas disponible.\n"
            "\n"

            "RÉPONSE ABSENTE :\n"
            "- Si absolument aucun extrait ne permet de répondre, "
            "répondez exactement : "
            "\"Je n’ai pas cette information pour le moment.\"\n"
            "\n"

            "STYLE :\n"
            "- Réponse naturelle et directe.\n"
            "- Ton poli et professionnel.\n"
            "- Réponse courte lorsque la question est simple.\n"
            "- Utilisez des listes uniquement lorsqu'elles améliorent la clarté.\n"
            "- Évitez les répétitions.\n"
            "- Ne mentionnez jamais les mots "
            "\"documents\", \"sources\", \"contexte\" ou \"extraits\" "
            "dans votre réponse finale.\n"
            "- Répondez comme un membre de l'équipe de l'Institut ILC.\n"
        )

    # =========================================================
    # USER PROMPT
    # =========================================================

    @classmethod
    def build_user_prompt(
        cls,
        user_question: str,
        context: str,
    ) -> str:

        cleaned_question = cls._clean_text(
            user_question
        )

        if context:
            return (
                "QUESTION À TRAITER :\n"
                f"{cleaned_question}\n"
                "\n"
                "INFORMATIONS AUTORISÉES :\n"
                f"{context}\n"
                "\n"
                "Répondez directement à la question en utilisant "
                "uniquement les informations autorisées ci-dessus. "
                "N'ajoutez aucune information extérieure."
            )

        return (
            "QUESTION À TRAITER :\n"
            f"{cleaned_question}\n"
            "\n"
            "Aucune information exploitable n'est disponible.\n"
            "\n"
            "Répondez exactement : "
            "Je n’ai pas cette information pour le moment."
        )

    # =========================================================
    # PROMPT FINAL
    # =========================================================

    @classmethod
    def build_prompt(
        cls,
        user_question: str,
        documents: list[dict[str, Any]],
        max_documents: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Construit les messages envoyés au modèle de chat.
        """

        context = cls.build_context(
            documents=documents,
            max_documents=max_documents,
        )

        system_prompt = (
            cls.build_system_prompt()
        )

        user_prompt = (
            cls.build_user_prompt(
                user_question=user_question,
                context=context,
            )
        )

        logger.info(
            "Prompt construit avec contexte=%s",
            bool(context),
        )

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]