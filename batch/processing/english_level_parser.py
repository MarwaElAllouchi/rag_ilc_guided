from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from docx import Document


class EnglishLevelParser:
    """
    Parse le document niveau_programme_anglais.docx
    et extrait les niveaux anglais avec leur tranche d'âge.
    """

    @staticmethod
    def parse_levels(file_path: Path) -> dict[str, Any]:
        document = Document(file_path)

        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        levels: list[dict[str, Any]] = []

        current_level: str | None = None

        for text in paragraphs:

            # LEVEL 1, LEVEL 2, ...
            level_match = re.fullmatch(
                r"LEVEL\s+(\d+)",
                text,
                flags=re.IGNORECASE,
            )

            if level_match:
                current_level = f"Level {level_match.group(1)}"
                continue

            # Âge indicatif : 3 à 4 ans
            age_match = re.search(
                r"âge\s+indicatif\s*:\s*(\d+)\s*(?:à|-)\s*(\d+)\s*ans?",
                text,
                flags=re.IGNORECASE,
            )

            if age_match and current_level:

                levels.append(
                    {
                        "level": current_level,
                        "age_min": int(age_match.group(1)),
                        "age_max": int(age_match.group(2)),
                    }
                )

                current_level = None

        return {
            "levels": levels
        }