from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class BusinessRulesEngine:
    """
    Moteur de règles métier pour les données structurées de l'Institut ILC :

    - tarifs
    - niveaux du programme arabe
    - cycles du programme arabe
    - niveaux du programme anglais

    Ces réponses sont construites directement depuis les fichiers JSON métier
    et ne dépendent pas du LLM.
    """

    def __init__(self) -> None:
        settings = get_settings()

        self.business_data_path = settings.business_data_path

        self._tarifs_cache: list[dict[str, str]] | None = None
        self._cycles_niveaux_cache: dict[str, Any] | None = None
        self._english_levels_cache: dict[str, Any] | None = None

    # =========================================================
    # UTILITAIRES TEXTE
    # =========================================================

    @staticmethod
    def _clean_text(value: Any) -> str:
        """
        Nettoie une valeur texte simple.
        """

        if value is None:
            return ""

        text = str(value).strip()

        if not text:
            return ""

        return text

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """
        Normalise un texte pour les comparaisons simples.
        """

        return (
            BusinessRulesEngine
            ._clean_text(value)
            .lower()
        )

    # =========================================================
    # CHARGEMENT JSON
    # =========================================================

    def _load_json(
        self,
        file_name: str,
    ) -> Any:
        """
        Charge un fichier JSON métier.
        """

        file_path = (
            self.business_data_path
            / file_name
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Fichier business introuvable : {file_path}"
            )

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    # =========================================================
    # TARIFS
    # =========================================================

    def get_all_tarifs(
        self,
        force_reload: bool = False,
    ) -> list[dict[str, str]]:
        """
        Charge les tarifs depuis le JSON métier,
        avec cache mémoire simple.
        """

        if (
            self._tarifs_cache is None
            or force_reload
        ):
            data = self._load_json(
                "formules_tarifs.json"
            )

            if not isinstance(data, list):
                raise ValueError(
                    "formules_tarifs.json doit contenir une liste."
                )

            self._tarifs_cache = data

        logger.info(
            "BusinessRulesEngine : %s tarifs chargés depuis JSON",
            len(self._tarifs_cache),
        )

        return self._tarifs_cache

    # =========================================================
    # NIVEAUX ARABE
    # =========================================================

    def get_cycles_niveaux_data(
        self,
        force_reload: bool = False,
    ) -> dict[str, Any]:
        """
        Charge les niveaux et cycles du programme arabe
        depuis le JSON métier.
        """

        if (
            self._cycles_niveaux_cache is None
            or force_reload
        ):
            data = self._load_json(
                "cycles_niveaux.json"
            )

            if not isinstance(data, dict):
                raise ValueError(
                    "cycles_niveaux.json doit contenir un objet JSON."
                )

            self._cycles_niveaux_cache = data

        logger.info(
            "BusinessRulesEngine : %s niveaux et %s cycles chargés depuis JSON",
            len(
                self._cycles_niveaux_cache.get(
                    "niveaux_par_annee",
                    [],
                )
            ),
            len(
                self._cycles_niveaux_cache.get(
                    "cycles",
                    [],
                )
            ),
        )

        return self._cycles_niveaux_cache

    # =========================================================
    # NIVEAUX ANGLAIS
    # =========================================================

    def get_english_levels_data(
        self,
        force_reload: bool = False,
    ) -> dict[str, Any]:
        """
        Charge les niveaux du programme anglais
        depuis le JSON métier.
        """

        if (
            self._english_levels_cache is None
            or force_reload
        ):
            data = self._load_json(
                "niveaux_anglais.json"
            )

            if not isinstance(data, dict):
                raise ValueError(
                    "niveaux_anglais.json doit contenir un objet JSON."
                )

            self._english_levels_cache = data

        logger.info(
            "BusinessRulesEngine : %s niveaux anglais chargés depuis JSON",
            len(
                self._english_levels_cache.get(
                    "levels",
                    [],
                )
            ),
        )

        return self._english_levels_cache

    # =========================================================
    # TRI TARIFS
    # =========================================================

    @staticmethod
    def _sort_tarifs(
        tarifs: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Trie les tarifs par prix croissant si possible.
        """

        def extract_price(
            item: dict[str, str],
        ) -> float:

            raw_price = (
                str(item.get("tarif", ""))
                .replace(",", ".")
                .strip()
            )

            raw_price = re.sub(
                r"[^\d.]",
                "",
                raw_price,
            )

            try:
                return float(raw_price)

            except Exception:
                return 999999.0

        return sorted(
            tarifs,
            key=extract_price,
        )

    # =========================================================
    # EXTRACTION ANNÉE DE NAISSANCE
    # =========================================================

    @staticmethod
    def _extract_birth_year(
        question: str,
    ) -> str | None:
        """
        Extrait une année de naissance de type 20xx.
        """

        match = re.search(
            r"\b(20\d{2})\b",
            question,
        )

        return (
            match.group(1)
            if match
            else None
        )

    # =========================================================
    # EXTRACTION ÂGE
    # =========================================================

    @staticmethod
    def _extract_age(
        question: str,
    ) -> int | None:
        """
        Extrait l'âge depuis :

        - "8 ans"
        - "8 an"
        - "8ans"
        - "8"
        """

        text = (
            question
            .lower()
            .strip()
        )

        match = re.search(
            r"\b(\d{1,2})\s*ans?\b",
            text,
        )

        if match:
            age = int(
                match.group(1)
            )

            if 0 <= age <= 99:
                return age

        if re.fullmatch(
            r"\d{1,2}",
            text,
        ):
            age = int(text)

            if 0 <= age <= 99:
                return age

        return None

    # =========================================================
    # EXTRACTION NIVEAU ARABE
    # =========================================================

    @staticmethod
    def _extract_level_name(
        question: str,
    ) -> str | None:
        """
        Détecte un niveau arabe explicite dans la question.

        Exemples :
        - NDA
        - CP1
        - CP2
        - N1
        - N2A
        - N3B
        - BARAÉM / BARAEM / BAREAM
        - JE
        """

        q = question.lower()

        patterns = [
            r"\bnda\b",
            r"\bn[1-7][ab]?\b",
            r"\bcp1\b",
            r"\bcp2\b",
            r"\bbaream\b",
            r"\bbaraem\b",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                q,
            )

            if match:
                value = (
                    match
                    .group(0)
                    .upper()
                )

                if value in {
                    "BARAEM",
                    "BAREAM",
                }:
                    return "BARAÉM"

                return value

        # Cas particulier : "niveau JE"
        if re.search(
            r"\bniveau\s+je\b",
            q,
        ):
            return "JE"

        # Cas : "JE"
        if re.search(
            r"\bJE\b",
            question,
        ):
            return "JE"

        return None

    # =========================================================
    # EXTRACTION CYCLE ARABE
    # =========================================================

    @staticmethod
    def _extract_cycle_name(
        question: str,
    ) -> str | None:
        """
        Détecte un cycle cité dans la question.

        Accepte les deux formats :

        - Cycle I
        - Cycle II
        - Cycle III
        - Cycle IV
        - Cycle V
        - Cycle VI

        ainsi que :

        - Cycle 1
        - Cycle 2
        - Cycle 3
        - Cycle 4
        - Cycle 5
        - Cycle 6
        """

        q = question.lower()

        match = re.search(
            r"\bcycle\s+(i|ii|iii|iv|v|vi|[1-6])\b",
            q,
        )

        if not match:
            return None

        value = (
            match
            .group(1)
            .upper()
        )

        number_to_roman = {
            "1": "I",
            "2": "II",
            "3": "III",
            "4": "IV",
            "5": "V",
            "6": "VI",
        }

        roman = number_to_roman.get(
            value,
            value,
        )

        return f"CYCLE {roman}"

    # =========================================================
    # NORMALISATION NIVEAU
    # =========================================================

    @staticmethod
    def _normalize_level_name(
        value: str,
    ) -> str:
        """
        Normalise un nom de niveau arabe.
        """

        normalized = (
            value
            .strip()
            .upper()
        )

        if normalized in {
            "BARAEM",
            "BAREAM",
            "BARAÉM",
        }:
            return "BARAÉM"

        return normalized

    # =========================================================
    # RECHERCHE NIVEAU PAR ANNÉE
    # =========================================================

    def _find_niveau_by_birth_year(
        self,
        birth_year: str,
    ) -> dict[str, str] | None:
        """
        Trouve un niveau arabe à partir
        d'une année de naissance.

        Gère aussi les formats comme :
        "2014/2013"
        """

        data = (
            self.get_cycles_niveaux_data()
        )

        niveaux = data.get(
            "niveaux_par_annee",
            [],
        )

        for item in niveaux:

            annee_value = self._clean_text(
                item.get(
                    "annee",
                    "",
                )
            )

            if not annee_value:
                continue

            if birth_year == annee_value:
                return item

            split_values = [
                value.strip()
                for value
                in annee_value.split("/")
                if value.strip()
            ]

            if birth_year in split_values:
                return item

        return None

    # =========================================================
    # RECHERCHE NIVEAU PAR NOM
    # =========================================================

    def _find_niveau_by_name(
        self,
        level_name: str,
    ) -> dict[str, str] | None:
        """
        Trouve un niveau arabe à partir
        de son nom explicite.
        """

        data = (
            self.get_cycles_niveaux_data()
        )

        niveaux = data.get(
            "niveaux_par_annee",
            [],
        )

        normalized_target = (
            self._normalize_level_name(
                level_name
            )
        )

        for item in niveaux:

            item_level = (
                self._normalize_level_name(
                    item.get(
                        "niveau",
                        "",
                    )
                )
            )

            if item_level == normalized_target:
                return item

        return None

    # =========================================================
    # RECHERCHE CYCLE
    # =========================================================

    def _find_cycle_by_name(
        self,
        cycle_name: str,
    ) -> dict[str, Any] | None:
        """
        Trouve un cycle arabe à partir
        de son nom.
        """

        data = (
            self.get_cycles_niveaux_data()
        )

        cycles = data.get(
            "cycles",
            [],
        )

        target = (
            self._normalize_text(
                cycle_name
            )
        )

        for item in cycles:

            item_name = (
                self._normalize_text(
                    item.get(
                        "cycle_name",
                        "",
                    )
                )
            )

            if item_name == target:
                return item

        return None

    # =========================================================
    # RÉPONSE TARIFS GLOBAUX
    # =========================================================

    def build_tarif_answer(
        self,
    ) -> str:
        """
        Construit la liste globale des tarifs,
        regroupée par catégorie.
        """

        tarifs = self.get_all_tarifs()

        tarifs = self._sort_tarifs(
            tarifs
        )

        if not tarifs:
            return (
                "Je suis désolé, je ne dispose pas actuellement "
                "des informations tarifaires."
            )

        lines = [
            "Voici les tarifs actuellement disponibles "
            "à l’Institut Langues et Cultures :",
            "",
        ]

        current_category = None

        for item in tarifs:

            categorie = self._clean_text(
                item.get(
                    "categorie",
                    "",
                )
            )

            formule = self._clean_text(
                item.get(
                    "formule",
                    "",
                )
            )

            duree = self._clean_text(
                item.get(
                    "duree",
                    "",
                )
            )

            prise_en_charge = self._clean_text(
                item.get(
                    "prise_en_charge_pause_dejeuner",
                    "",
                )
            )

            tarif = self._clean_text(
                item.get(
                    "tarif",
                    "",
                )
            )

            # Affichage du titre de catégorie
            if categorie and categorie != current_category:
                if current_category is not None:
                    lines.append("")

                lines.append(
                    f"### {categorie}"
                )
                lines.append("")

                current_category = categorie

            # Formule
            line = f"- **{formule}**"

            if duree:
                line += f" — {duree}"

            line += f" : **{tarif}**"

            # Information supplémentaire pour les formules journée
            if (
                prise_en_charge
                and prise_en_charge.lower() != "non concerné"
            ):
                line += (
                    f" — {prise_en_charge} "
                    "de la pause déjeuner"
                )

            lines.append(line)

        lines.append("")
        lines.append(
            "N’hésitez pas à préciser la formule qui vous intéresse "
            "si vous souhaitez davantage de détails."
        )

        return "\n".join(lines)
    
    #==========================================
    # Tarif  pour une catégorie précise
    #==========================================

    def build_tarif_category_answer(
    self,
    categorie: str,
    ) -> str | None:
        """
        Construit une réponse tarifaire pour une catégorie précise.

        Exemples :
        - Cours & demi-journée
        - Formules journée
        - Accueil
        """

        tarifs = self.get_all_tarifs()

        if not tarifs:
            return None

        categorie_normalisee = self._normalize_text(categorie)

        matches = [
            item
            for item in tarifs
            if self._normalize_text(
                item.get("categorie", "")
            ) == categorie_normalisee
        ]

        if not matches:
            return None

        matches = self._sort_tarifs(matches)

        lines = [
            f"### {categorie}",
            "",
        ]

        for item in matches:

            formule = self._clean_text(
                item.get("formule", "")
            )

            duree = self._clean_text(
                item.get("duree", "")
            )

            tarif = self._clean_text(
                item.get("tarif", "")
            )

            prise_en_charge = self._clean_text(
                item.get(
                    "prise_en_charge_pause_dejeuner",
                    "",
                )
            )

            line = f"- **{formule}**"

            if duree:
                line += f" — {duree}"

            line += f" : **{tarif}**"

            if (
                prise_en_charge
                and prise_en_charge.lower() != "non concerné"
            ):
                line += (
                    f" — {prise_en_charge} "
                    "de la pause déjeuner"
                )

            lines.append(line)

        return "\n".join(lines)

    # =========================================================
    # RÉPONSE TARIF SPÉCIFIQUE
    # =========================================================

    def build_tarif_specific_answer(
    self,
    question: str,
    ) -> str | None:
        """
        Construit une réponse ciblée sur une formule tarifaire précise.

        La correspondance privilégie les noms complets de formules
        afin d'éviter de mélanger plusieurs tarifs proches.
        """

        tarifs = self.get_all_tarifs()

        if not tarifs:
            return None

        q = self._normalize_text(question)

        matches: list[dict[str, str]] = []

        for item in tarifs:

            formule = self._normalize_text(
                item.get(
                    "formule",
                    "",
                )
            )

            if not formule:
                continue

            # Correspondance stricte sur le nom complet de la formule
            if formule in q:
                matches.append(item)

        if not matches:
            return None

        matches = self._sort_tarifs(matches)

        # Une seule ligne tarifaire trouvée
        if len(matches) == 1:

            item = matches[0]

            formule = self._clean_text(
                item.get("formule", "")
            )

            duree = self._clean_text(
                item.get("duree", "")
            )

            tarif = self._clean_text(
                item.get("tarif", "")
            )

            prise_en_charge = self._clean_text(
                item.get(
                    "prise_en_charge_pause_dejeuner",
                    "",
                )
            )

            lines = [
                f"Voici l’information disponible pour la formule **{formule}** :",
                "",
            ]

            if duree:
                lines.append(
                    f"- Horaires / durée : **{duree}**"
                )

            lines.append(
                f"- Tarif : **{tarif}**"
            )

            if (
                prise_en_charge
                and prise_en_charge.lower() != "non concerné"
            ):
                lines.append(
                    f"- Pause déjeuner : **{prise_en_charge}**"
                )

            return "\n".join(lines)

        # Plusieurs variantes de la même formule
        lines = [
            "Voici les tarifs correspondant à votre demande :",
            "",
        ]

        for item in matches:

            formule = self._clean_text(
                item.get("formule", "")
            )

            duree = self._clean_text(
                item.get("duree", "")
            )

            tarif = self._clean_text(
                item.get("tarif", "")
            )

            prise_en_charge = self._clean_text(
                item.get(
                    "prise_en_charge_pause_dejeuner",
                    "",
                )
            )

            line = f"- **{formule}**"

            if duree:
                line += f" — {duree}"

            line += f" : **{tarif}**"

            if (
                prise_en_charge
                and prise_en_charge.lower() != "non concerné"
            ):
                line += f" — {prise_en_charge} de la pause déjeuner"

            lines.append(line)

        return "\n".join(lines)

    # =========================================================
    # NIVEAU ARABE PAR ÂGE / ANNÉE DE NAISSANCE
    # =========================================================

    def build_niveau_birth_year_answer(
        self,
        question: str,
    ) -> dict[str, Any] | None:
        """
        Répond à partir :

        - de l'année de naissance
        - ou de l'âge

        Retourne également le ou les cycles trouvés.
        """

        birth_year = (
            self._extract_birth_year(
                question
            )
        )

        # -----------------------------------------------------
        # Aucun année donnée → essayer avec l'âge
        # -----------------------------------------------------

        if not birth_year:

            age = self._extract_age(
                question
            )

            if age is None:
                return None

            current_year = (
                datetime.now().year
            )

            possible_birth_years = [
                str(
                    current_year - age
                ),
                str(
                    current_year - age - 1
                ),
            ]

            matches = [
                self._find_niveau_by_birth_year(
                    year
                )
                for year
                in possible_birth_years
            ]

            matches = [
                item
                for item
                in matches
                if item
            ]

            if not matches:
                return None

            cycles = {
                self._clean_text(
                    item.get(
                        "cycle_name",
                        "",
                    )
                )
                for item
                in matches
                if self._clean_text(
                    item.get(
                        "cycle_name",
                        "",
                    )
                )
            }

            # -------------------------------------------------
            # Les deux années possibles donnent le même cycle
            # -------------------------------------------------

            if len(cycles) == 1:

                cycle_name = next(
                    iter(cycles)
                )

                return {
                    "answer": (
                        f"D’après l’âge indiqué (**{age} ans**), "
                        f"votre enfant correspond au **{cycle_name}**. "
                        "Le niveau précis dépend de son année de naissance."
                    ),
                    "selected_cycles": [
                        cycle_name
                    ],
                }

            # -------------------------------------------------
            # Plusieurs cycles possibles
            # -------------------------------------------------

            return {
                "answer": (
                    f"À **{age} ans**, le niveau dépend de l’année "
                    "de naissance de votre enfant. "
                    "Pouvez-vous m’indiquer son année de naissance ?"
                ),
                "selected_cycles": [],
            }

        # -----------------------------------------------------
        # Année de naissance fournie
        # -----------------------------------------------------

        niveau_item = (
            self._find_niveau_by_birth_year(
                birth_year
            )
        )

        if not niveau_item:
            return {
                "answer": (
                    "Je suis désolé, je n’ai pas trouvé de niveau "
                    "correspondant à l’année de naissance "
                    f"**{birth_year}** dans les informations disponibles."
                ),
                "selected_cycles": [],
            }

        niveau = self._clean_text(
            niveau_item.get(
                "niveau",
                "",
            )
        )

        cycle_name = self._clean_text(
            niveau_item.get(
                "cycle_name",
                "",
            )
        )

        if cycle_name:
            return {
                "answer": (
                    "D’après les informations disponibles, "
                    f"un élève né en **{birth_year}** correspond "
                    f"au **niveau {niveau}**, rattaché au "
                    f"**{cycle_name}**."
                ),
                "selected_cycles": [
                    cycle_name
                ],
            }

        return {
            "answer": (
                "D’après les informations disponibles, "
                f"un élève né en **{birth_year}** correspond "
                f"au **niveau {niveau}**."
            ),
            "selected_cycles": [],
        }

    # =========================================================
    # NIVEAU ANGLAIS PAR ÂGE
    # =========================================================

    def build_english_level_by_age_answer(
        self,
        question: str,
    ) -> dict[str, Any] | None:
        """
        Oriente vers un ou plusieurs niveaux d'anglais
        uniquement à partir de l'âge indiqué.

        Retourne également les niveaux sélectionnés.
        """

        age = self._extract_age(
            question
        )

        if age is None:
            return None

        data = (
            self.get_english_levels_data()
        )

        levels = data.get(
            "levels",
            [],
        )

        matches: list[str] = []

        for item in levels:

            level = self._clean_text(
                item.get(
                    "level",
                    "",
                )
            )

            age_min = item.get(
                "age_min"
            )

            age_max = item.get(
                "age_max"
            )

            if not level:
                continue

            if (
                not isinstance(
                    age_min,
                    int,
                )
                or not isinstance(
                    age_max,
                    int,
                )
            ):
                continue

            if age_min <= age <= age_max:
                matches.append(
                    level
                )

        # -----------------------------------------------------
        # Aucun niveau
        # -----------------------------------------------------

        if not matches:
            return {
                "answer": (
                    "Je n’ai pas de niveau d’anglais correspondant "
                    f"précisément à **{age} ans** dans les informations "
                    "disponibles."
                ),
                "selected_levels": [],
            }

        # -----------------------------------------------------
        # Un seul niveau
        # -----------------------------------------------------

        if len(matches) == 1:

            selected_level = (
                matches[0]
            )

            return {
                "answer": (
                    f"D’après l’âge indiqué (**{age} ans**), "
                    f"votre enfant correspond au "
                    f"**{selected_level}** du programme d’anglais."
                ),
                "selected_levels": [
                    selected_level
                ],
            }

        # -----------------------------------------------------
        # Plusieurs niveaux possibles
        # -----------------------------------------------------

        levels_text = " ou ".join(
            f"**{level}**"
            for level
            in matches
        )

        return {
            "answer": (
                f"Pour **{age} ans**, la grille disponible indique "
                f"{levels_text} pour le programme d’anglais. "
                "L’âge seul ne permet donc pas de départager précisément "
                "ces niveaux."
            ),
            "selected_levels": matches,
        }

    # =========================================================
    # NIVEAU ARABE EXPLICITE
    # =========================================================

    def build_niveau_specific_answer(
        self,
        question: str,
    ) -> str | None:
        """
        Retourne une réponse directe sur un niveau arabe explicite :

        - année de naissance
        - cycle associé
        - description du cycle
        """

        level_name = (
            self._extract_level_name(
                question
            )
        )

        if not level_name:
            return None

        niveau_item = (
            self._find_niveau_by_name(
                level_name
            )
        )

        if not niveau_item:
            return None

        niveau = self._clean_text(
            niveau_item.get(
                "niveau",
                "",
            )
        )

        annee = self._clean_text(
            niveau_item.get(
                "annee",
                "",
            )
        )

        cycle_name = self._clean_text(
            niveau_item.get(
                "cycle_name",
                "",
            )
        )

        cycle_item = (
            self._find_cycle_by_name(
                cycle_name
            )
            if cycle_name
            else None
        )

        cycle_description = (
            self._clean_text(
                cycle_item.get(
                    "description",
                    "",
                )
            )
            if cycle_item
            else ""
        )

        lines = [
            (
                "D’après les informations disponibles, "
                f"le **niveau {niveau}** correspond "
                f"aux élèves nés en **{annee}**."
            )
        ]

        if cycle_name:
            lines.append(
                f"Ce niveau est rattaché au **{cycle_name}**."
            )

        if cycle_description:
            lines.append("")

            lines.append(
                f"**Description du {cycle_name} :**"
            )

            lines.append(
                cycle_description
            )

        return "\n".join(lines)

    # =========================================================
    # CYCLE ARABE EXPLICITE
    # =========================================================

    def build_cycle_answer(
        self,
        question: str,
    ) -> str | None:
        """
        Retourne directement les informations
        d'un cycle arabe explicite.
        """

        cycle_name = (
            self._extract_cycle_name(
                question
            )
        )

        if not cycle_name:
            return None

        cycle_item = (
            self._find_cycle_by_name(
                cycle_name
            )
        )

        if not cycle_item:
            return None

        description = self._clean_text(
            cycle_item.get(
                "description",
                "",
            )
        )

        related_levels = (
            cycle_item.get(
                "related_levels",
                [],
            )
        )

        lines = [
            (
                "Voici les informations disponibles "
                f"pour le **{cycle_name}** :"
            )
        ]

        if related_levels:

            lines.append("")

            lines.append(
                "**Niveaux associés :** "
                + ", ".join(
                    related_levels
                )
            )

        if description:

            lines.append("")

            lines.append(
                "**Description :**"
            )

            lines.append(
                description
            )

        return "\n".join(lines)

    # =========================================================
    # DÉTECTION QUESTION TARIFS GLOBALE
    # =========================================================

    @staticmethod
    def is_tarif_global_question(
        question: str,
    ) -> bool:
        """
        Détecte les formulations globales
        de demande de tarifs.
        """

        q = question.lower()

        global_patterns = [
            "quels sont les tarifs",
            "quels sont vos tarifs",
            "donne moi les tarifs",
            "donnez moi les tarifs",
            "donnez-moi les tarifs",
            "liste des tarifs",
            "tous les tarifs",
            "les tarifs",
            "combien coûtent les cours",
            "combien coutent les cours",
        ]

        short_forms = {
            "prix",
            "tarif",
            "tarifs",
        }

        if q.strip() in short_forms:
            return True

        return any(
            pattern in q
            for pattern
            in global_patterns
        )