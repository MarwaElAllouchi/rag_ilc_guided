from __future__ import annotations

import logging
import re
from typing import Any

from app.core.config import get_settings
from app.rag.business_rules import BusinessRulesEngine
from app.rag.pipeline import RagPipeline
from app.rag.query_analyzer import QueryAnalyzer
from app.rag.small_talk_detector import SmallTalkDetector
from app.rag.guided_choices import (
    MAIN_CHOICES,
    LEVEL_PROGRAM_CHOICES,
    TARIF_CHOICES,
)

logger = logging.getLogger(__name__)

INSCRIPTION_CHOICES = [
    {"label": "📝 Comment s’inscrire ?", "value": "inscription_comment"},
    {"label": "💳 Paiement", "value": "inscription_paiement"},
    {"label": "← Retour", "value": "menu_principal"},
]


MYSCOL_CHOICES = [
    {"label": "📚 Notes & bulletins", "value": "myscol_notes_bulletins"},
    {"label": "👨‍👩‍👧 Aide MyScol", "value": "myscol_aide"},
    {"label": "← Retour", "value": "menu_principal"},
]


class RagRouter:
    """
    Router guidé autour de 4 besoins principaux du chatbot ILC :
    tarifs, niveau, inscription et MyScol.

    Les questions libres concernant les horaires restent prises en charge.

    La saisie libre reste disponible mais ne déclenche plus de RAG général.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.pipeline = RagPipeline()
        self.business_engine = BusinessRulesEngine()
        self.small_talk_detector = SmallTalkDetector()

    @staticmethod
    def _clean_question(question: str | None) -> str:
        if not isinstance(question, str):
            return ""
        return question.strip()

    @staticmethod
    def _build_response(
        answer: str,
        documents: list[dict[str, Any]] | None = None,
        intent: str = "general",
        route: str = "unknown",
        suggestions: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "documents": documents or [],
            "intent": intent,
            "route": route,
            "suggestions": suggestions or [],
        }

    def get_main_suggestions(self) -> list[dict[str, str]]:
        return MAIN_CHOICES

    @staticmethod
    def _is_english_reference(question: str) -> bool:
        q = question.lower()
        return bool(
            re.search(r"\blevel\s*[1-4]\b", q)
            or "anglais" in q
            or "english" in q
        )

    @staticmethod
    def _is_arabic_reference(question: str) -> bool:
        q = question.lower()
        return bool(
            re.search(r"\b(cp\s*[12]|n\s*[1-7][ab]?|nda|baraem|baream)\b", q)
            or "arabe" in q
            or re.search(r"\bcycle\s+(i|ii|iii|iv|v|vi|\d+)\b", q)
        )
    @staticmethod
    def _contains_specific_arabic_level_reference(question: str) -> bool:
        """
        Détecte une référence précise à un niveau ou cycle arabe.

        Exemples :
        - CP1
        - CP 2
        - N1
        - N 3A
        - NDA
        - Baraem
        - Cycle I
        - Cycle 2
        """
        q = question.lower()

        return bool(
            re.search(
                r"\b("
                r"cp\s*[12]"
                r"|n\s*[1-7][ab]?"
                r"|nda"
                r"|baraem"
                r"|baream"
                r"|cycle\s+(?:i|ii|iii|iv|v|vi|\d+)"
                r")\b",
                q,
            )
        )

    def _detect_program_reference(self, question: str) -> str | None:
        """
        Détecte le programme uniquement à partir de références explicites.

        Retourne :
        - "arabe" si la question contient une référence arabe uniquement ;
        - "anglais" si la question contient une référence anglaise uniquement ;
        - None si aucun programme n'est identifiable ou si la question est ambiguë.

        Cette méthode évite de coder un programme en dur dans les parcours d'inscription.
        """
        is_arabic = self._is_arabic_reference(question)
        is_english = self._is_english_reference(question)

        if is_arabic and not is_english:
            return "arabe"

        if is_english and not is_arabic:
            return "anglais"

        return None

    @staticmethod
    def _get_session_value(session: Any | None, field: str) -> Any | None:
        """Lit une valeur de session depuis un modèle Pydantic ou un dictionnaire."""
        if session is None:
            return None

        if isinstance(session, dict):
            return session.get(field)

        return getattr(session, field, None)

    def _get_session_program(self, session: Any | None) -> str | None:
        """Retourne uniquement un programme de session valide."""
        program = self._get_session_value(session, "program")

        if not isinstance(program, str):
            return None

        program = program.strip().lower()
        if program in {"arabe", "anglais"}:
            return program

        return None

    def _build_known_child_context(
        self,
        question: str,
        session: Any | None,
    ) -> tuple[int | None, int | None, str | None]:
        """
        Construit le contexte structuré connu sur l'enfant.

        Le programme cité dans la question est prioritaire sur le programme mémorisé.
        """
        child_age = self._get_session_value(session, "child_age")
        child_birth_year = self._get_session_value(session, "child_birth_year")

        if not isinstance(child_age, int):
            child_age = None

        if not isinstance(child_birth_year, int):
            child_birth_year = None

        explicit_program = self._detect_program_reference(question)
        program = explicit_program or self._get_session_program(session)

        return child_age, child_birth_year, program

    @staticmethod
    def _contains_schedule_topic(question: str) -> bool:
        q = question.lower()
        keywords = (
            "horaire", "horaires", "heure", "heures", "jour de cours",
            "jours de cours", "calendrier", "ouverture", "fermeture",
            "accueil matin", "accueil soir",
        )
        return any(keyword in q for keyword in keywords)

    @staticmethod
    def _contains_myscol_topic(question: str) -> bool:
        q = question.lower()
        keywords = ("myscol", "bulletin", "bulletins", "note", "notes", "suivi")
        return any(keyword in q for keyword in keywords)

    def _detect_supported_topic(self, question: str) -> str | None:
        if self._contains_myscol_topic(question):
            return "myscol"
        if self._contains_schedule_topic(question):
            return "horaires"

        intent = QueryAnalyzer.detect_intent(question)
        if intent in {"tarif", "niveau", "inscription"}:
            return intent
        return None

    def route_choice(self, choice: str, top_k: int | None = None) -> dict[str, Any]:
        cleaned_choice = self._clean_question(choice).lower()
        logger.info("Router → choix guidé : %s", cleaned_choice)

        effective_top_k = top_k if top_k is not None else self.settings.top_k
        if effective_top_k <= 0:
            raise ValueError("top_k doit être strictement positif")

        if cleaned_choice == "menu_principal":
            return self._build_response(
                answer="Comment puis-je vous aider ?",
                intent="menu",
                route="guided_main_menu",
                suggestions=MAIN_CHOICES,
            )
        
        if cleaned_choice == "menu_tarifs":
            return self._build_response(
                answer="Choisissez la catégorie de tarifs qui vous intéresse :",
                intent="tarif",
                route="guided_tarifs_menu",
                suggestions=TARIF_CHOICES,
            )

        if cleaned_choice == "tarifs_cours":
            answer = self.business_engine.build_tarif_category_answer(
                "Cours & demi-journée"
            )

            return self._build_response(
                answer=answer or "Je n’ai pas cette information pour le moment.",
                intent="tarif",
                route="guided_tarifs_cours",
                suggestions=TARIF_CHOICES,
            )

        if cleaned_choice == "tarifs_journee":
            answer = self.business_engine.build_tarif_category_answer(
                "Formules journée"
            )

            return self._build_response(
                answer=answer or "Je n’ai pas cette information pour le moment.",
                intent="tarif",
                route="guided_tarifs_journee",
                suggestions=TARIF_CHOICES,
            )

        if cleaned_choice == "tarifs_accueil":
            answer = self.business_engine.build_tarif_category_answer(
                "Accueil"
            )

            return self._build_response(
                answer=answer or "Je n’ai pas cette information pour le moment.",
                intent="tarif",
                route="guided_tarifs_accueil",
                suggestions=TARIF_CHOICES,
            )

        if cleaned_choice == "menu_niveaux":
            return self._build_response(
                answer="Pour quel programme souhaitez-vous trouver le niveau de votre enfant ?",
                intent="niveau",
                route="guided_niveau_programme",
                suggestions=LEVEL_PROGRAM_CHOICES,
            )

        if cleaned_choice == "niveau_programme_arabe":
            return self._build_response(
                answer="Quel âge a votre enfant ? Vous pouvez aussi indiquer son année de naissance.",
                intent="niveau",
                route="guided_niveau_arabe_age",
                suggestions=[{"label": "← Retour", "value": "menu_niveaux"}],
            )

        if cleaned_choice == "niveau_programme_anglais":
            return self._build_response(
                answer="Quel âge a votre enfant ?",
                intent="niveau",
                route="guided_niveau_anglais_age",
                suggestions=[{"label": "← Retour", "value": "menu_niveaux"}],
            )

        if cleaned_choice.startswith("arabe_cycle_"):
            cycle_number = cleaned_choice.removeprefix("arabe_cycle_")
            if cycle_number.isdigit():
                rag_question = (
                    f"Programme d'arabe Cycle {cycle_number} : quels sont les niveaux concernés "
                    f"et quelles sont les informations pédagogiques relatives au Cycle {cycle_number} ?"
                )
                result = self.pipeline.run(
                    user_question=rag_question,
                    top_k=effective_top_k,
                    intent="niveau",
                    categories=["niveau"],
                    programs=["arabe"],
                )
                return self._build_response(
                    answer=result["answer"],
                    documents=result.get("documents", []),
                    intent="niveau",
                    route="guided_arabe_cycle_rag",
                    suggestions=MAIN_CHOICES,
                )

        if cleaned_choice.startswith("anglais_level_"):
            level_number = cleaned_choice.removeprefix("anglais_level_")
            if level_number.isdigit():
                rag_question = (
                    f"Programme d'anglais Level {level_number} : quel est l'âge indicatif, "
                    f"l'objectif pédagogique et le contenu du Level {level_number} ?"
                )
                result = self.pipeline.run(
                    user_question=rag_question,
                    top_k=effective_top_k,
                    intent="niveau",
                    categories=["niveau"],
                    programs=["anglais"],
                )
                return self._build_response(
                    answer=result["answer"],
                    documents=result.get("documents", []),
                    intent="niveau",
                    route="guided_anglais_level_rag",
                    suggestions=MAIN_CHOICES,
                )

        if cleaned_choice == "menu_inscription":
            return self._build_response(
                answer="Que souhaitez-vous savoir sur l’inscription ?",
                intent="inscription",
                route="guided_inscription_menu",
                suggestions=INSCRIPTION_CHOICES,
            )

        if cleaned_choice == "inscription_comment":
            return self._answer_inscription_general(effective_top_k)

        if cleaned_choice == "inscription_niveau_precis":
            return self._build_response(
                answer=(
                    "Indiquez le niveau dans lequel vous souhaitez inscrire votre enfant "
                    "(par exemple : CP1, N2, Cycle II ou Level 2)."
                ),
                intent="inscription",
                route="guided_inscription_niveau_precis_waiting_level",
                suggestions=[{"label": "← Retour", "value": "menu_inscription"}],
            )

        if cleaned_choice == "inscription_paiement":
            return self._answer_inscription_paiement(effective_top_k)

        if cleaned_choice == "menu_myscol":
            return self._build_response(
                answer="Que souhaitez-vous savoir sur MyScol ?",
                intent="myscol",
                route="guided_myscol_menu",
                suggestions=MYSCOL_CHOICES,
            )

        if cleaned_choice == "myscol_notes_bulletins":
            return self._answer_myscol(
                "Où et comment les parents peuvent-ils consulter les notes et les bulletins "
                "de leur enfant sur MyScol ?",
                effective_top_k,
                "guided_myscol_notes_bulletins_rag",
            )

        if cleaned_choice == "myscol_aide":
            return self._answer_myscol(
                "Quelles informations utiles sont disponibles pour les parents concernant "
                "l'utilisation de MyScol et le suivi de leur enfant ?",
                effective_top_k,
                "guided_myscol_aide_rag",
            )

        return self._build_response(
            answer=(
                "Ce choix n’est plus disponible. Je peux vous aider avec les tarifs, "
                "le niveau de votre enfant, l’inscription, les horaires ou MyScol."
            ),
            intent="unknown_choice",
            route="guided_unknown_choice",
            suggestions=MAIN_CHOICES,
        )

    def route(
        self,
        question: str,
        top_k: int | None = None,
        context: str | None = None,
        session: Any | None = None,
    ) -> dict[str, Any]:
        cleaned_question = self._clean_question(question)
        cleaned_context = self._clean_question(context).lower() if context else ""

        if not cleaned_question:
            return self._build_response(
                answer="Écrivez votre question ou choisissez un sujet ci-dessous.",
                intent="empty",
                route="empty_input",
                suggestions=MAIN_CHOICES,
            )

        effective_top_k = top_k if top_k is not None else self.settings.top_k
        if effective_top_k <= 0:
            raise ValueError("top_k doit être strictement positif")

        small_talk = self.small_talk_detector.detect(cleaned_question)

        if small_talk.kind == "greeting":
            return self._build_response(
                answer=(
                    "Bonjour 👋 Je suis l’assistant virtuel de l’Institut ILC. "
                    "Je peux vous aider avec les tarifs, le niveau de votre enfant, "
                    "l’inscription, les horaires et MyScol."
                ),
                intent="greeting",
                route="small_talk_greeting",
                suggestions=MAIN_CHOICES,
            )

        if small_talk.kind == "gratitude":
            return self._build_response(
                answer="Avec plaisir 😊",
                intent="gratitude",
                route="small_talk_gratitude",
                suggestions=MAIN_CHOICES,
            )

        if small_talk.kind == "farewell":
            return self._build_response(
                answer="Au revoir 👋 À bientôt.",
                intent="farewell",
                route="small_talk_farewell",
                suggestions=MAIN_CHOICES,
            )

        if small_talk.kind in {"acknowledgment", "praise"}:
            return self._build_response(
                answer="Avec plaisir 😊 Que souhaitez-vous consulter ?",
                intent=small_talk.kind,
                route=f"small_talk_{small_talk.kind}",
                suggestions=MAIN_CHOICES,
            )

        if getattr(small_talk, "cleaned_text", ""):
            cleaned_question = self._clean_question(small_talk.cleaned_text)

        if not cleaned_question:
            return self._build_response(
                answer="Que souhaitez-vous consulter ?",
                intent="empty_after_small_talk_cleaning",
                route="empty_after_small_talk_cleaning",
                suggestions=MAIN_CHOICES,
            )

        if cleaned_context == "niveau_arabe":
            return self._handle_niveau_arabe_guided(cleaned_question)

        if cleaned_context == "niveau_anglais":
            return self._handle_niveau_anglais_guided(cleaned_question)

        topic = self._detect_supported_topic(cleaned_question)
        logger.info("Router → thème supporté détecté : %s", topic)

        if topic == "tarif":
            return self._handle_tarif(cleaned_question)
        if topic == "niveau":
            return self._handle_niveau_free(
                cleaned_question,
                effective_top_k,
                session=session,
            )
        if topic == "inscription":
            return self._handle_inscription_free(
                cleaned_question,
                effective_top_k,
                session=session,
            )
        if topic == "horaires":
            return self._answer_horaires(
                cleaned_question,
                effective_top_k,
                "free_horaires_rag",
            )
        if topic == "myscol":
            return self._answer_myscol(
                cleaned_question,
                effective_top_k,
                "free_myscol_rag",
            )

        return self._build_response(
            answer=(
                "Je peux vous aider concernant les tarifs, le niveau de votre enfant, "
                "l’inscription, les horaires et MyScol. Choisissez un sujet ci-dessous 👇"
            ),
            intent="unsupported",
            route="unsupported_free_question",
            suggestions=MAIN_CHOICES,
        )

    def _handle_tarif(self, question: str) -> dict[str, Any]:
        logger.info("Router → moteur métier TARIF")

        if self.business_engine.is_tarif_global_question(question):
            return self._build_response(
                answer=self.business_engine.build_tarif_answer(),
                intent="tarif",
                route="tarif_business_engine_global",
                suggestions=MAIN_CHOICES,
            )

        specific_answer = self.business_engine.build_tarif_specific_answer(question)
        if specific_answer:
            return self._build_response(
                answer=specific_answer,
                intent="tarif",
                route="tarif_business_engine_specific",
                suggestions=MAIN_CHOICES,
            )

        return self._build_response(
            answer=self.business_engine.build_tarif_answer(),
            intent="tarif",
            route="tarif_business_engine_fallback",
            suggestions=MAIN_CHOICES,
        )

    def _handle_niveau_arabe_guided(self, question: str) -> dict[str, Any]:
        logger.info("Router → parcours guidé NIVEAU ARABE : %s", question)

        birth_year_result = self.business_engine.build_niveau_birth_year_answer(question)

        if birth_year_result:
            if isinstance(birth_year_result, dict):
                answer = birth_year_result.get("answer", "")
                selected_cycles = birth_year_result.get("selected_cycles", [])
                suggestions: list[dict[str, str]] = []

                for cycle_name in selected_cycles:
                    cycle_value = cycle_name.lower().replace("cycle", "").strip()
                    roman_to_number = {
                        "i": "1", "ii": "2", "iii": "3",
                        "iv": "4", "v": "5", "vi": "6",
                    }
                    cycle_number = roman_to_number.get(cycle_value, cycle_value)
                    if cycle_number.isdigit():
                        suggestions.append({
                            "label": f"📘 Voir les détails du {cycle_name}",
                            "value": f"arabe_cycle_{cycle_number}",
                        })

                suggestions.extend(MAIN_CHOICES)
                return self._build_response(
                    answer=answer,
                    intent="niveau",
                    route="niveau_arabe_business_engine",
                    suggestions=suggestions,
                )

            return self._build_response(
                answer=str(birth_year_result),
                intent="niveau",
                route="niveau_arabe_business_engine",
                suggestions=MAIN_CHOICES,
            )

        niveau_answer = self.business_engine.build_niveau_specific_answer(question)
        if niveau_answer:
            return self._build_response(
                answer=niveau_answer,
                intent="niveau",
                route="niveau_arabe_business_engine_direct",
                suggestions=MAIN_CHOICES,
            )

        cycle_answer = self.business_engine.build_cycle_answer(question)
        if cycle_answer:
            return self._build_response(
                answer=cycle_answer,
                intent="niveau",
                route="cycle_arabe_business_engine_direct",
                suggestions=MAIN_CHOICES,
            )

        return self._build_response(
            answer=(
                "Pour trouver le niveau d’arabe, indiquez simplement l’âge de votre enfant "
                "ou son année de naissance."
            ),
            intent="niveau",
            route="niveau_arabe_need_age",
            suggestions=[{"label": "← Retour", "value": "menu_niveaux"}],
        )

    def _handle_niveau_anglais_guided(self, question: str) -> dict[str, Any]:
        logger.info("Router → parcours guidé NIVEAU ANGLAIS : %s", question)

        result = self.business_engine.build_english_level_by_age_answer(question)
        if not result:
            return self._build_response(
                answer="Pour trouver le niveau d’anglais, indiquez simplement l’âge de votre enfant.",
                intent="niveau",
                route="niveau_anglais_need_age",
                suggestions=[{"label": "← Retour", "value": "menu_niveaux"}],
            )

        suggestions: list[dict[str, str]] = []
        for level in result.get("selected_levels", []):
            level_number = level.lower().replace("level", "").strip()
            if level_number.isdigit():
                suggestions.append({
                    "label": f"📘 Voir les détails du {level}",
                    "value": f"anglais_level_{level_number}",
                })
        suggestions.extend(MAIN_CHOICES)

        return self._build_response(
            answer=result["answer"],
            intent="niveau",
            route="niveau_anglais_business_engine",
            suggestions=suggestions,
        )

    def _handle_niveau_free(
        self,
        question: str,
        top_k: int,
        session: Any | None = None,
    ) -> dict[str, Any]:
        explicit_program = self._detect_program_reference(question)
        session_program = self._get_session_program(session)
        program = explicit_program or session_program

        child_age = self._get_session_value(session, "child_age")
        child_birth_year = self._get_session_value(session, "child_birth_year")

        if not isinstance(child_age, int):
            child_age = None

        if not isinstance(child_birth_year, int):
            child_birth_year = None

        if program == "anglais":
            age_question = question

            if child_age is not None:
                age_question = f"{question}\nL'enfant a {child_age} ans."
            elif child_birth_year is not None:
                age_question = f"{question}\nL'enfant est né en {child_birth_year}."

            english_result = self.business_engine.build_english_level_by_age_answer(
                age_question
            )

            if english_result:
                suggestions: list[dict[str, str]] = []

                for level in english_result.get("selected_levels", []):
                    level_number = level.lower().replace("level", "").strip()

                    if level_number.isdigit():
                        suggestions.append({
                            "label": f"📘 Voir les détails du {level}",
                            "value": f"anglais_level_{level_number}",
                        })

                suggestions.extend(MAIN_CHOICES)

                return self._build_response(
                    answer=english_result["answer"],
                    intent="niveau",
                    route="free_niveau_anglais_business",
                    suggestions=suggestions,
                )

            result = self.pipeline.run(
                user_question=question,
                top_k=top_k,
                intent="niveau",
                categories=["niveau"],
                programs=["anglais"],
            )

            return self._build_response(
                answer=result["answer"],
                documents=result.get("documents", []),
                intent="niveau",
                route="free_niveau_anglais_rag",
                suggestions=MAIN_CHOICES,
            )

        if program == "arabe":
           # Si le parent mentionne explicitement un niveau/cycle,
            # on conserve sa question telle quelle.
            if self._contains_specific_arabic_level_reference(question):
                return self._handle_niveau_arabe_guided(question)

            # Sinon, pour une recherche générale de niveau,
            # on peut utiliser les informations mémorisées en session.
            guided_question = question

            if child_age is not None:
                guided_question = str(child_age)
            elif child_birth_year is not None:
                guided_question = str(child_birth_year)

            return self._handle_niveau_arabe_guided(guided_question)

        return self._build_response(
            answer="Pour quel programme souhaitez-vous trouver le niveau de votre enfant ?",
            intent="niveau",
            route="free_niveau_choose_program",
            suggestions=LEVEL_PROGRAM_CHOICES,
        )

    def _answer_inscription_general(self, top_k: int) -> dict[str, Any]:
        question = (
            "Comment inscrire un enfant à l'Institut ILC ? Réponds uniquement avec les principales "
            "modalités d'inscription disponibles dans les documents. N'ajoute aucune condition "
            "qui n'est pas explicitement indiquée."
        )
        result = self.pipeline.run(
            user_question=question,
            top_k=top_k,
            intent="inscription",
            categories=["inscription"],
            programs=["general"],
        )
        return self._build_response(
            answer=result["answer"],
            documents=result.get("documents", []),
            intent="inscription",
            route="guided_inscription_general_rag",
            suggestions=INSCRIPTION_CHOICES,
        )

    def _answer_inscription_niveau_precis(
        self,
        question: str,
        top_k: int,
        session: Any | None = None,
    ) -> dict[str, Any]:
        """
        Répond à une question d'inscription concernant un niveau précis.

        Le programme cité dans la question est prioritaire. Sinon, le programme
        mémorisé dans la session peut être utilisé. L'âge et l'année de naissance
        mémorisés sont utilisés uniquement pour les questions de niveau/admission.
        """
        child_age, child_birth_year, program = self._build_known_child_context(
            question=question,
            session=session,
        )

        if program is None:
            return self._build_response(
                answer=(
                    "Pour quel programme souhaitez-vous inscrire votre enfant ? "
                    "Indiquez le niveau concerné, par exemple CP1 ou N2 pour l’arabe, "
                    "ou Level 1 / Level 2 pour l’anglais."
                ),
                intent="inscription",
                route="inscription_niveau_program_unknown",
                suggestions=LEVEL_PROGRAM_CHOICES,
            )

        known_context: list[str] = []

        if child_age is not None:
            known_context.append(
                f"Âge déjà indiqué par le parent : {child_age} ans."
            )

        if child_birth_year is not None:
            known_context.append(
                f"Année de naissance déjà indiquée : {child_birth_year}."
            )

        known_context.append(f"Programme concerné : {program}.")

        context_text = "\n".join(known_context)

        rag_question = (
            f"Question du parent : {question}\n"
            f"{context_text}\n"
            "Réponds uniquement avec les informations explicitement disponibles "
            "dans les documents de l'Institut ILC concernant ce niveau et son admission. "
            "Prends en compte l'âge ou l'année de naissance déjà connus lorsqu'ils sont fournis. "
            "Ne mélange jamais les programmes arabe et anglais. "
            "Ne commence jamais par « Oui, vous pouvez » si l'admission n'est pas "
            "explicitement confirmée par les informations disponibles. "
            "Ne déduis jamais qu'un âge correspond à un niveau précis si cette correspondance "
            "n'est pas explicitement indiquée dans les documents. "
            "Si l'âge exact requis pour le niveau demandé n'est pas disponible, dis-le clairement. "
            "Ne confirme pas l'admission uniquement à partir de l'âge ou du nom du cycle. "
            "Si l'âge connu paraît incompatible avec le niveau demandé selon les documents, "
            "explique-le clairement. "
            "Si les documents ne permettent pas de confirmer directement l'admission "
            "dans ce niveau, dis-le clairement et n'invente aucune règle."
        )

        result = self.pipeline.run(
            user_question=rag_question,
            top_k=top_k,
            intent="niveau",
            categories=["niveau"],
            programs=[program],
        )

        return self._build_response(
            answer=result["answer"],
            documents=result.get("documents", []),
            intent="inscription",
            route=f"inscription_niveau_precis_{program}_rag",
            suggestions=INSCRIPTION_CHOICES,
        )

    def _answer_inscription_paiement(self, top_k: int) -> dict[str, Any]:
        question = (
            "Quels sont les moyens de paiement disponibles à l'Institut ILC et peut-on payer en plusieurs fois ? "
            "Réponds uniquement à partir des informations explicites disponibles dans les documents."
        )
        result = self.pipeline.run(
            user_question=question,
            top_k=top_k,
            intent="inscription",
            categories=["inscription", "paiement"],
            programs=["general"],
        )
        return self._build_response(
            answer=result["answer"],
            documents=result.get("documents", []),
            intent="inscription",
            route="guided_inscription_paiement_rag",
            suggestions=INSCRIPTION_CHOICES,
        )

    def _handle_inscription_free(
        self,
        question: str,
        top_k: int,
        session: Any | None = None,
    ) -> dict[str, Any]:
        q = question.lower()

        if (
            QueryAnalyzer.contains_niveau_pattern(question)
            or self._detect_program_reference(question) is not None
        ):
            return self._answer_inscription_niveau_precis(
                question=question,
                top_k=top_k,
                session=session,
            )

        if any(word in q for word in (
            "payer", "paiement", "virement", "chèque", "cheque",
            "espèces", "especes", "plusieurs fois",
        )):
            return self._answer_inscription_paiement(top_k)

        return self._answer_inscription_general(top_k)

    def _answer_horaires(self, question: str, top_k: int, route: str) -> dict[str, Any]:
        result = self.pipeline.run(
            user_question=question,
            top_k=top_k,
            intent="general",
            programs=["general"],
        )
        return self._build_response(
            answer=result["answer"],
            documents=result.get("documents", []),
            intent="horaires",
            route=route,
            suggestions=MAIN_CHOICES,
        )

    def _answer_myscol(self, question: str, top_k: int, route: str) -> dict[str, Any]:
        result = self.pipeline.run(
            user_question=question,
            top_k=top_k,
            intent="general",
            programs=["general"],
            source_files=["guide_myscol_bulletins.pdf", "MyScol.docx"],
        )
        return self._build_response(
            answer=result["answer"],
            documents=result.get("documents", []),
            intent="myscol",
            route=route,
            suggestions=MYSCOL_CHOICES,
        )
