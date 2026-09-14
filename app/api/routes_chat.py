from __future__ import annotations

import logging

from fastapi import APIRouter

from app.api.schemas import ChatRequest, ChatResponse
from app.rag.router import RagRouter
from batch.embeddings.embedder import EmbeddingServiceUnavailableError


logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])
rag_router = RagRouter()


class ChatMessages:
    """
    Messages standardisés de l'API chat.
    """

    EMPTY_QUESTION = (
        "Bonjour 👋 Je suis l’assistant virtuel de l’Institut ILC. "
        "Je peux vous aider concernant les niveaux, les tarifs, l’inscription "
        "et les informations générales de l’institut. Comment puis-je vous aider ?"
    )

    SEARCH_UNAVAILABLE = (
        "Le service de recherche est momentanément indisponible. "
        "Merci de réessayer dans quelques instants."
    )

    TECHNICAL_ERROR = (
        "Une difficulté technique est survenue. "
        "Merci de réessayer dans quelques instants."
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Envoyer une question au chatbot",
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Endpoint principal de conversation avec l'assistant ILC.
    """

    question = request.question.strip() if request.question else ""
    choice = request.choice.strip() if request.choice else ""
    context = request.context.strip() if request.context else ""

    session = request.session

    logger.info(
        (
            "Requête /chat reçue "
            "(question_vide=%s, choice=%s, context=%s, "
            "child_age=%s, child_birth_year=%s, program=%s, top_k=%s)"
        ),
        not bool(question),
        choice or None,
        context or None,
        session.child_age if session else None,
        session.child_birth_year if session else None,
        session.program if session else None,
        request.top_k,
    )

    # -------------------------------------------------------
    # OUVERTURE DU CHATBOT
    # -------------------------------------------------------

    if not question and not choice:
        return ChatResponse(
            answer=ChatMessages.EMPTY_QUESTION,
            intent="empty",
            route="empty_input",
            documents=[],
            suggestions=rag_router.get_main_suggestions(),
        )

    try:

        # -------------------------------------------------------
        # CHOIX PROVENANT D'UN BOUTON
        # -------------------------------------------------------

        if choice:
            result = rag_router.route_choice(
                choice=choice,
                top_k=request.top_k,
            )

        # -------------------------------------------------------
        # QUESTION LIBRE DE L'UTILISATEUR
        # -------------------------------------------------------

        else:
            result = rag_router.route(
                question=question,
                top_k=request.top_k,
                context=context or None,
                session=session,
            )

        return ChatResponse(
            answer=result.get("answer", "") or ChatMessages.TECHNICAL_ERROR,
            intent=result.get("intent"),
            route=result.get("route"),
            documents=result.get("documents", []),
            suggestions=result.get("suggestions", []),
        )

    except EmbeddingServiceUnavailableError:

        logger.warning(
            "Service d'embedding indisponible pour la requête /chat"
        )

        return ChatResponse(
            answer=ChatMessages.SEARCH_UNAVAILABLE,
            intent="technical_error",
            route="embedding_unavailable",
            documents=[],
            suggestions=[],
        )

    except Exception:

        logger.exception(
            "Erreur inattendue lors du traitement de la requête /chat"
        )

        return ChatResponse(
            answer=ChatMessages.TECHNICAL_ERROR,
            intent="technical_error",
            route="unexpected_error",
            documents=[],
            suggestions=[],
        )