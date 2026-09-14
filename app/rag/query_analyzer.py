from __future__ import annotations

import logging
import re


logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Détecte uniquement les intentions utiles au chatbot ILC :

    - tarif
    - niveau
    - inscription
    - general

    Les thèmes horaires et MyScol sont détectés directement
    dans router.py.
    """

    TARIF_KEYWORDS = {
        "tarif",
        "tarifs",
        "prix",
        "cout",
        "coût",
        "combien",
        "formule",
        "formules",
    }

    NIVEAU_KEYWORDS = {
        "niveau",
        "niveaux",
        "classe",
        "âge",
        "age",
        "année",
        "annee",
        "né",
        "nee",
        "nés",
        "nees",
        "cycle",
        "cycles",
    }

    INSCRIPTION_KEYWORDS = {
        "inscription",
        "inscrire",
        "inscrit",
        "inscris",
        "inscrite",
        "inscrits",
        "inscrites",
        "s'inscrire",
        "m'inscrire",
        "l'inscrire",
        "dossier",
        "documents",
        "secrétariat",
        "secretariat",
        "admission",
        "payer",
        "paiement",
        "plusieurs fois",
        "virement",
        "chèque",
        "cheque",
        "espèces",
        "especes",
    }

    SALUTATION_KEYWORDS = {
        "bonjour",
        "salut",
        "coucou",
        "hello",
        "bonsoir",
        "cc",
        "slt",
        "hi",
    }

    LEVEL_PATTERNS = [
        r"\bn\s*[1-7][ab]?\b",
        r"\bcp\s*[12]\b",
        r"\bnda\b",
        r"\bbaream\b",
        r"\bbaraem\b",
        r"\blevel\s*[1-4]\b",
    ]

    EXTRA_NIVEAU_PATTERNS = [
        r"\bniveau\s+[1-7][ab]?\b",
        r"\bcycle\s+(i|ii|iii|iv|v|vi|\d+)\b",
        r"\bniveau\s+je\b",
        r"\btest\s+de\s+niveau\b",
    ]

    @staticmethod
    def normalize_question(question: str) -> str:
        if not isinstance(question, str):
            return ""

        q = question.strip().lower()
        q = re.sub(r"\s+", " ", q)

        return q

    @staticmethod
    def clean_for_keywords(question: str) -> str:
        q = QueryAnalyzer.normalize_question(question)

        q = re.sub(
            r"[?!.,;:()\"'\-]+",
            " ",
            q,
        )

        q = re.sub(
            r"\s+",
            " ",
            q,
        ).strip()

        return q

    @staticmethod
    def _contains_any_keyword(
        question: str,
        keywords: set[str],
    ) -> bool:

        cleaned = QueryAnalyzer.clean_for_keywords(question)
        words = set(cleaned.split())

        for keyword in keywords:
            keyword_cleaned = (
                QueryAnalyzer.clean_for_keywords(keyword)
            )

            if " " in keyword_cleaned:
                if keyword_cleaned in cleaned:
                    return True

            elif keyword_cleaned in words:
                return True

        return False

    # =========================================================
    # TARIFS
    # =========================================================

    @staticmethod
    def contains_tarif_pattern(
        question: str,
    ) -> bool:

        return QueryAnalyzer._contains_any_keyword(
            question,
            QueryAnalyzer.TARIF_KEYWORDS,
        )

    # =========================================================
    # NIVEAUX
    # =========================================================

    @staticmethod
    def contains_niveau_pattern(
        question: str,
    ) -> bool:

        q = QueryAnalyzer.normalize_question(question)

        if QueryAnalyzer._contains_any_keyword(
            q,
            QueryAnalyzer.NIVEAU_KEYWORDS,
        ):
            return True

        for pattern in (
            QueryAnalyzer.LEVEL_PATTERNS
            + QueryAnalyzer.EXTRA_NIVEAU_PATTERNS
        ):
            if re.search(
                pattern,
                q,
                flags=re.IGNORECASE,
            ):
                return True

        return False

    # =========================================================
    # INSCRIPTION
    # =========================================================

    @staticmethod
    def contains_inscription_pattern(
        question: str,
    ) -> bool:

        return QueryAnalyzer._contains_any_keyword(
            question,
            QueryAnalyzer.INSCRIPTION_KEYWORDS,
        )

    # =========================================================
    # SALUTATIONS
    # =========================================================

    @staticmethod
    def is_greeting(
        question: str,
    ) -> bool:

        q = QueryAnalyzer.normalize_question(question)

        return any(
            q.startswith(word)
            for word in QueryAnalyzer.SALUTATION_KEYWORDS
        )

    @staticmethod
    def strip_greeting(
        question: str,
    ) -> str:

        q = question.strip()

        greeting_patterns = [
            (
                r"^(bonjour|salut|coucou|hello|bonsoir|cc|slt|hi)"
                r"\b[\s,;:!?-]*"
            ),
        ]

        cleaned = q

        for pattern in greeting_patterns:
            cleaned = re.sub(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

        return cleaned

    # =========================================================
    # OUTILS GÉNÉRAUX
    # =========================================================

    @staticmethod
    def get_business_keywords() -> set[str]:

        return (
            QueryAnalyzer.TARIF_KEYWORDS
            | QueryAnalyzer.NIVEAU_KEYWORDS
            | QueryAnalyzer.INSCRIPTION_KEYWORDS
        )

    @staticmethod
    def is_short_ambiguous_question(
        question: str,
    ) -> bool:

        cleaned = QueryAnalyzer.clean_for_keywords(question)

        if not cleaned:
            return True

        words = cleaned.split()

        if len(words) == 1:
            return True

        if (
            len(words) == 2
            and all(
                word in QueryAnalyzer.get_business_keywords()
                for word in words
            )
        ):
            return True

        return False

    @staticmethod
    def is_valid_question(
        question: str,
    ) -> bool:

        q = QueryAnalyzer.normalize_question(question)

        if not q:
            return False

        if QueryAnalyzer.is_greeting(q):
            return True

        cleaned = QueryAnalyzer.clean_for_keywords(q)
        words = cleaned.split()

        if len(words) >= 2:
            return True

        if QueryAnalyzer.contains_inscription_pattern(q):
            return True

        if QueryAnalyzer.contains_tarif_pattern(q):
            return True

        if QueryAnalyzer.contains_niveau_pattern(q):
            return True

        return False

    # =========================================================
    # DÉTECTION D'INTENTION
    # =========================================================

    @staticmethod
    def detect_intent(
        question: str,
    ) -> str:
        """
        Retourne :

        - tarif
        - niveau
        - inscription
        - general
        """

        q = QueryAnalyzer.normalize_question(question)

        if not q:
            logger.info(
                "Intention détectée : general (question vide)"
            )
            return "general"

        # Une demande d'inscription est prioritaire.
        if QueryAnalyzer.contains_inscription_pattern(q):
            logger.info(
                "Intention détectée : inscription"
            )
            return "inscription"

        if QueryAnalyzer.contains_tarif_pattern(q):
            logger.info(
                "Intention détectée : tarif"
            )
            return "tarif"

        if QueryAnalyzer.contains_niveau_pattern(q):
            logger.info(
                "Intention détectée : niveau"
            )
            return "niveau"

        logger.info(
            "Intention détectée : general"
        )

        return "general"