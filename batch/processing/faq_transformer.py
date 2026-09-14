from __future__ import annotations

import logging
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


class FaqTransformer:
    """
    Transforme une FAQ tabulaire en sorties hybrides :
    - business
    - rag
    - rejected

    Logique cible :
    - toute FAQ complète enrichit le RAG
    - certaines FAQ complètes alimentent aussi le moteur business
    - seules les FAQ incomplètes (ou explicitement rejetées) sont rejetées
    """

    COLUMN_ALIASES = {
        "id": ["id", "faq_id"],
        "question": ["question", "questions"],
        "answer": ["answer", "reponse", "réponse"],
        "category": ["category", "categorie", "catégorie", "categorie_optionnel"],
        "status": ["status", "statut"],
        "target": ["target", "cible"],
        "keywords": ["keywords", "mots_cles", "mots_clés"],
    }

    BUSINESS_CATEGORIES = {
        "inscription",
        "tarif",
        "paiement",
        "niveau",
        "myscol",
        "contact",
        "horaires",
    }

    @staticmethod
    def _clean_value(value: Any) -> str:
        if pd.isna(value):
            return ""

        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "null"}:
            return ""

        return text

    @classmethod
    def _clean_row(cls, row: pd.Series, columns: list[str]) -> dict[str, str]:
        return {
            col: cls._clean_value(row.get(col, ""))
            for col in columns
        }

    @staticmethod
    def _resolve_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
        for alias in aliases:
            if alias in df.columns:
                return alias
        return None

    @classmethod
    def _resolve_columns(cls, df: pd.DataFrame) -> dict[str, str]:
        resolved: dict[str, str] = {}

        for logical_name, aliases in cls.COLUMN_ALIASES.items():
            found = cls._resolve_column(df, aliases)
            if found:
                resolved[logical_name] = found

        if "question" not in resolved or "answer" not in resolved:
            raise ValueError(
                "Colonnes FAQ obligatoires manquantes : question et/ou answer/reponse. "
                f"Colonnes disponibles : {sorted(df.columns.tolist())}"
            )

        return resolved

    @staticmethod
    def _classify_category(question: str, answer: str = "") -> str:
        text = f"{question} {answer}".lower()

        if any(word in text for word in ["tarif", "prix", "coût", "cout", "combien"]):
            return "tarif"

        if any(word in text for word in ["payer", "paiement", "virement", "chèque", "cheque", "especes", "espèces"]):
            return "paiement"

        if any(word in text for word in ["niveau", "âge", "age", "ans", "cycle", "classe", "test de niveau"]):
            return "niveau"

        if any(word in text for word in ["inscription", "inscrire", "dossier", "documents"]):
            return "inscription"

        if any(word in text for word in ["myscol", "application", "code école", "code ecole"]):
            return "myscol"

        if any(word in text for word in ["horaire", "ouverture", "vacances", "matin", "garderie"]):
            return "horaires"

        if any(word in text for word in ["parents", "bulletin", "enseignants", "réunion", "reunion", "communiquer"]):
            return "communication"

        if any(word in text for word in ["activité", "activite", "sortie", "événement", "evenement", "sport", "arts"]):
            return "activites"

        if any(word in text for word in ["programme", "suivez-vous", "progrès", "progression", "difficultés", "difficultes"]):
            return "pedagogie"

        if any(word in text for word in ["reglement", "règlement", "interieur", "intérieur", "autorisé", "interdit"]):
            return "reglement"

        return "general"
    
    @staticmethod
    def _detect_program(question: str, answer: str = "") -> str:
        text = f"{question} {answer}".lower()

        if "arabe" in text or "arabic" in text:
            return "arabe"

        if "anglais" in text or "english" in text:
            return "anglais"

        return "general"


    @staticmethod
    def _normalize_target(value: str) -> str:
        return value.strip().lower()

    @classmethod
    def _should_add_to_business(cls, category: str, explicit_target: str = "") -> bool:
        """
        Détermine si la FAQ doit aussi alimenter le moteur business.
        """
        if explicit_target == "business":
            return True

        if explicit_target == "rag":
            return False

        return category in cls.BUSINESS_CATEGORIES

    @staticmethod
    def _should_add_to_rag(explicit_target: str = "") -> bool:
        """
        Toute FAQ complète va dans le RAG sauf rejet explicite.
        """
        return explicit_target != "reject"

    @staticmethod
    def _build_business_record(
        faq_id: str,
        question: str,
        answer: str,
        category: str,
        keywords: str,
        source_file: str,
    ) -> dict[str, Any]:
        return {
            "id": faq_id,
            "question": question,
            "answer": answer,
            "category": category,
            "keywords": keywords,
            "source_file": source_file,
        }

    @staticmethod
    def _build_rag_document(
        faq_id: str,
        question: str,
        answer: str,
        category: str,
        program: str,
        keywords: str,
        source_file: str,
        language: str,
    ) -> dict[str, Any]:
        return {
            "content": f"Question : {question}\nRéponse : {answer}",
            "metadata": {
                "faq_id": faq_id,
                "source_type": "faq",
                "category": category,
                "program": program,
                "keywords": keywords,
                "source_file": source_file,
                "language": language,
            },
        }

    @staticmethod
    def _deduplicate_business_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, str]] = set()
        deduplicated: list[dict[str, Any]] = []

        for record in records:
            key = (
                str(record.get("question", "")).strip().lower(),
                str(record.get("answer", "")).strip().lower(),
                str(record.get("category", "")).strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(record)

        return deduplicated

    @staticmethod
    def _deduplicate_rag_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        deduplicated: list[dict[str, Any]] = []

        for doc in documents:
            content = str(doc.get("content", "")).strip()
            metadata = doc.get("metadata", {}) or {}
            category = str(metadata.get("category", "")).strip().lower()

            key = (content, category)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(doc)

        return deduplicated

    @classmethod
    def transform(
        cls,
        df: pd.DataFrame,
        source_file: str = "faq.xlsx",
        language: str = "fr",
    ) -> dict[str, Any]:
        resolved_columns = cls._resolve_columns(df)

        business_documents: list[dict[str, Any]] = []
        rag_documents: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []

        for idx, row in df.iterrows():
            row_data = cls._clean_row(row, list(df.columns))

            question = row_data.get(resolved_columns["question"], "")
            answer = row_data.get(resolved_columns["answer"], "")

            faq_id = row_data.get(resolved_columns["id"], "") if "id" in resolved_columns else ""
            category = row_data.get(resolved_columns["category"], "") if "category" in resolved_columns else ""
            explicit_target = row_data.get(resolved_columns["target"], "") if "target" in resolved_columns else ""
            keywords = row_data.get(resolved_columns["keywords"], "") if "keywords" in resolved_columns else ""

            if not faq_id:
                faq_id = f"faq_{idx + 1}"

            if not question or not answer:
                rejected_rows.append(
                    {
                        "row_index": idx,
                        "faq_id": faq_id,
                        "question": question,
                        "answer": answer,
                        "reason": "missing_question_or_answer",
                    }
                )
                continue

            if not category:
                category = cls._classify_category(question, answer)

            program = cls._detect_program(question, answer)

            explicit_target = cls._normalize_target(explicit_target)

            if explicit_target == "reject":
                rejected_rows.append(
                    {
                        "row_index": idx,
                        "faq_id": faq_id,
                        "question": question,
                        "answer": answer,
                        "reason": "explicit_reject",
                    }
                )
                continue

            if cls._should_add_to_business(category, explicit_target):
                business_documents.append(
                    cls._build_business_record(
                        faq_id=faq_id,
                        question=question,
                        answer=answer,
                        category=category,
                        keywords=keywords,
                        source_file=source_file,
                    )
                )

            if cls._should_add_to_rag(explicit_target):
                rag_documents.append(
                    cls._build_rag_document(
                        faq_id=faq_id,
                        question=question,
                        answer=answer,
                        category=category,
                        program=program,
                        keywords=keywords,
                        source_file=source_file,
                        language=language,
                    )
                )

        business_documents = cls._deduplicate_business_records(business_documents)
        rag_documents = cls._deduplicate_rag_documents(rag_documents)

        stats = {
            "total_rows": len(df),
            "business_count": len(business_documents),
            "rag_count": len(rag_documents),
            "rejected_count": len(rejected_rows),
        }

        logger.info(
            "FaqTransformer : %s business, %s rag, %s rejected",
            stats["business_count"],
            stats["rag_count"],
            stats["rejected_count"],
        )

        return {
            "business_documents": business_documents,
            "rag_documents": rag_documents,
            "rejected_rows": rejected_rows,
            "stats": stats,
        }