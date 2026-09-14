from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from batch.ingestion.file_loader import FileLoader
from batch.processing.chunker import TextChunker


logger = logging.getLogger(__name__)


class DocumentTransformer:
    """
    Transforme un document long (TXT / DOCX / PDF)
    en chunks prêts pour le RAG, avec métadonnées standardisées.

    Cette version :
    - conserve l'interface actuelle ;
    - conserve la structure de sortie actuelle ;
    - normalise les métadonnées ;
    - permet une catégorisation plus précise des chunks du règlement intérieur.
    """

    MIN_PARAGRAPH_LENGTH = 3
    SECTION_TITLE_MAX_WORDS = 12

    # Lorsqu'un nouveau titre de section apparaît très tard dans un chunk,
    # on conserve la catégorie précédente pour éviter de reclasser tout
    # le chunk à cause de quelques lignes de la section suivante.
    SECTION_SWITCH_MAX_RATIO = 0.40

    @staticmethod
    def transform_document(
        file_path: Path | str,
        source_type: str = "document",
        category: str = "general",
        program: str = "general",
        language: str = "fr",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Charge un document, le découpe en chunks
        et ajoute des métadonnées standard.

        Args:
            file_path: chemin du document source
            source_type: type métier de la source
            category: catégorie métier/logique par défaut
            program: programme associé au document
            language: langue du document
            chunk_size: taille cible des chunks
            chunk_overlap: chevauchement entre chunks

        Returns:
            Liste de documents chunkés avec contenu et métadonnées.
        """
        file_path = Path(file_path)

        source_type = source_type.strip().lower()
        category = category.strip().lower()
        program = program.strip().lower()
        language = language.strip().lower()

        DocumentTransformer._validate_inputs(
            file_path=file_path,
            source_type=source_type,
            program=program,
            category=category,
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        loader = FileLoader(file_path)
        content = loader.load()
        content = DocumentTransformer._clean_content(content)

        if not content:
            logger.warning(
                "Document ignoré car vide après nettoyage : %s",
                file_path,
            )
            return []

        prepared_content = DocumentTransformer._prepare_content_for_chunking(
            content
        )

        if not prepared_content:
            logger.warning(
                "Document ignoré car vide après préparation : %s",
                file_path,
            )
            return []

        chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = chunker.split_text(
            prepared_content
        )

        if not chunks:
            logger.warning(
                "Aucun chunk généré pour le document : %s",
                file_path,
            )
            return []

        total_chunks = len(chunks)
        document_id = DocumentTransformer._build_document_id(
            file_path
        )

        documents: list[dict[str, Any]] = []

        # Catégorie courante utilisée surtout pour les documents
        # contenant plusieurs sections métier, comme le règlement.
        current_category = category

        for idx, chunk in enumerate(chunks):

            section_title = (
                DocumentTransformer._extract_section_title_from_chunk(
                    chunk
                )
            )

            chunk_category = (
                DocumentTransformer._resolve_chunk_category(
                    file_path=file_path,
                    chunk=chunk,
                    default_category=category,
                    previous_category=current_category,
                    chunk_index=idx,
                )
            )

            current_category = chunk_category

            documents.append(
                {
                    "content": chunk,
                    "metadata": {
                        "document_id": document_id,
                        "source_type": source_type,
                        "category": chunk_category,
                        "language": language,
                        "program": program,
                        "source_file": file_path.name,
                        "source_path": str(file_path),
                        "document_title": file_path.stem,
                        "chunk_index": idx,
                        "chunk_size": len(chunk),
                        "total_chunks": total_chunks,
                        "is_chunked": total_chunks > 1,
                        "section_title": section_title,
                    },
                }
            )

        logger.info(
            "Document transformé : %s chunks générés pour %s",
            len(documents),
            file_path.name,
        )

        return documents

    # =========================================================
    # CATÉGORISATION MÉTIER DES CHUNKS
    # =========================================================

    @staticmethod
    def _resolve_chunk_category(
        file_path: Path,
        chunk: str,
        default_category: str,
        previous_category: str,
        chunk_index: int,
    ) -> str:
        """
        Détermine la catégorie métier d'un chunk.

        Cas particulier :
        reglement_interieur.docx contient plusieurs sujets :
        - inscription / admission
        - paiement / frais de scolarité
        - règlement scolaire

        Pour éviter que tout le document soit classé "reglement",
        cette méthode suit les sections du document.

        Si un nouveau titre apparaît trop tard dans le chunk,
        on conserve la catégorie précédente : le chunk appartient
        majoritairement encore à la section précédente.
        """
        file_stem = (
            file_path.stem.strip().lower()
        )

        if file_stem != "reglement_interieur":
            return default_category

        normalized = DocumentTransformer._normalize_for_matching(
            chunk
        )

        # Cas particulier du premier chunk du règlement :
        # l'introduction précède la section "INSCRIPTION ET ADMISSION",
        # mais le contenu métier principal reste l'inscription.
        if (
            chunk_index == 0
            and "inscription et admission" in normalized
        ):
            return "inscription"

        section_markers: list[
            tuple[int, str]
        ] = []

        patterns_by_category = {
            "inscription": [
                r"\b1\s*[-.)]?\s*inscription\s+et\s+admission\b",
                r"\binscription\s+et\s+admission\b",
            ],
            "paiement": [
                r"\b2\s*[-.)]?\s*frais\s+de\s+scolarite\b",
                r"\bfrais\s+de\s+scolarite\b",
            ],
            "reglement": [
                r"\b3\s*[-.)]?\s*obligations?\s+scolaires?\b",
                r"\b4\s*[-.)]?\s*absences?\s+et\s+retards?\b",
                r"\b5\s*[-.)]?\s*implication\b",
                r"\b6\s*[-.)]?\s*securite\b",
                r"\b7\s*[-.)]?\s*hygiene\s+et\s+sante\b",
                r"\b8\s*[-.)]?\s*periode\s+de\s+pandemie\b",
                r"\b9\s*[-.)]?\s*respect\s+et\s+adhesion\b",
            ],
        }

        for resolved_category, patterns in patterns_by_category.items():

            for pattern in patterns:
                match = re.search(
                    pattern,
                    normalized,
                    flags=re.IGNORECASE,
                )

                if match:
                    section_markers.append(
                        (
                            match.start(),
                            resolved_category,
                        )
                    )

        if not section_markers:
            return previous_category

        section_markers.sort(
            key=lambda item: item[0]
        )

        first_position, first_category = (
            section_markers[0]
        )

        chunk_length = max(
            len(normalized),
            1,
        )

        ratio = (
            first_position
            / chunk_length
        )

        # Si le nouveau titre apparaît assez tôt,
        # le chunk est rattaché à la nouvelle section.
        if (
            ratio
            <= DocumentTransformer.SECTION_SWITCH_MAX_RATIO
        ):
            return first_category

        # Sinon le chunk appartient encore principalement
        # à la section précédente.
        return previous_category

    @staticmethod
    def _normalize_for_matching(
        text: str,
    ) -> str:
        """
        Normalise un texte uniquement pour la détection
        de sections : minuscules, accents retirés et
        espaces harmonisés.
        """
        if not isinstance(
            text,
            str,
        ):
            return ""

        normalized = unicodedata.normalize(
            "NFKD",
            text,
        )

        normalized = "".join(
            char
            for char in normalized
            if not unicodedata.combining(
                char
            )
        )

        normalized = normalized.lower()

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_inputs(
        file_path: Path,
        source_type: str,
        category: str,
        program: str,
        language: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """
        Valide les entrées principales.
        """
        if not file_path.exists():
            raise FileNotFoundError(
                f"Document introuvable : {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Le chemin n'est pas un fichier valide : {file_path}"
            )

        if not source_type.strip():
            raise ValueError(
                "source_type ne peut pas être vide"
            )

        if not category.strip():
            raise ValueError(
                "category ne peut pas être vide"
            )

        if not program.strip():
            raise ValueError(
                "program ne peut pas être vide"
            )

        if not language.strip():
            raise ValueError(
                "language ne peut pas être vide"
            )

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size doit être strictement positif"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap ne peut pas être négatif"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap doit être inférieur à chunk_size"
            )

    # =========================================================
    # NETTOYAGE
    # =========================================================

    @staticmethod
    def _clean_content(
        content: str,
    ) -> str:
        """
        Nettoie le contenu brut du document.
        """
        if not isinstance(
            content,
            str,
        ):
            return ""

        return content.strip()

    @staticmethod
    def _build_document_id(
        file_path: Path,
    ) -> str:
        """
        Construit un identifiant simple et stable
        pour le document.
        """
        return (
            file_path
            .stem
            .lower()
            .replace(
                " ",
                "_",
            )
        )

    @staticmethod
    def _clean_paragraphs(
        content: str,
    ) -> list[str]:
        """
        Transforme le contenu
        en paragraphes propres.
        """
        raw_parts = re.split(
            r"\n+",
            content,
        )

        paragraphs: list[str] = []

        for part in raw_parts:

            cleaned = re.sub(
                r"\s+",
                " ",
                part,
            ).strip()

            if (
                len(cleaned)
                >= DocumentTransformer.MIN_PARAGRAPH_LENGTH
            ):
                paragraphs.append(
                    cleaned
                )

        return paragraphs

    # =========================================================
    # TITRES / SECTIONS
    # =========================================================

    @staticmethod
    def _is_probable_section_title(
        paragraph: str,
    ) -> bool:
        """
        Détecte les titres ou sections probables.

        La méthode reste volontairement simple
        pour conserver la compatibilité avec
        les documents existants.
        """
        text = paragraph.strip()

        if not text:
            return False

        words = text.split()

        if (
            len(words)
            > DocumentTransformer.SECTION_TITLE_MAX_WORDS
        ):
            return False

        if re.match(
            r"^\d+\s*[-.)]\s*",
            text,
        ):
            return True

        if text.endswith(
            ":"
        ):
            return True

        alpha_chars = [
            char
            for char in text
            if char.isalpha()
        ]

        if alpha_chars:

            uppercase_ratio = (
                sum(
                    1
                    for char in alpha_chars
                    if char.isupper()
                )
                / len(alpha_chars)
            )

            if (
                uppercase_ratio > 0.6
                and len(words) <= 8
            ):
                return True

        if (
            len(words) <= 6
            and text[:1].isupper()
        ):
            return True

        return False

    @staticmethod
    def _prepare_content_for_chunking(
        content: str,
    ) -> str:
        """
        Prépare le contenu pour le chunking :

        - nettoyage des paragraphes ;
        - conservation des titres ;
        - séparation claire des blocs.
        """
        paragraphs = (
            DocumentTransformer._clean_paragraphs(
                content
            )
        )

        if not paragraphs:
            return ""

        prepared_parts: list[str] = []

        for paragraph in paragraphs:

            if (
                DocumentTransformer._is_probable_section_title(
                    paragraph
                )
            ):
                section_title = (
                    paragraph.strip(
                        " :"
                    )
                )

                prepared_parts.append(
                    f"Section : {section_title}"
                )

            else:
                prepared_parts.append(
                    paragraph
                )

        return "\n\n".join(
            prepared_parts
        ).strip()

    @staticmethod
    def _extract_section_title_from_chunk(
        chunk: str,
    ) -> str | None:
        """
        Essaie d'extraire le premier titre de section
        présent au début du chunk préparé.
        """
        if (
            not isinstance(
                chunk,
                str,
            )
            or not chunk.strip()
        ):
            return None

        lines = [
            line.strip()
            for line in chunk.splitlines()
            if line.strip()
        ]

        if not lines:
            return None

        first_line = lines[0]

        if (
            first_line
            .lower()
            .startswith(
                "section :"
            )
        ):
            title = (
                first_line
                .split(
                    ":",
                    1,
                )[1]
                .strip()
            )

            return title or None

        return None
