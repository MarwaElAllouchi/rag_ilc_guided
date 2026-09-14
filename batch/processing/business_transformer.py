from __future__ import annotations

import logging
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


class BusinessTransformer:
    """
    Transforme les données tabulaires structurées en objets JSON métier
    exploitables par les moteurs déterministes.

    Objectifs :
    - nettoyage homogène des valeurs
    - validation des colonnes nécessaires
    - tolérance à certains synonymes de colonnes
    - rejet des lignes inexploitables
    - logs exploitables
    """

    FORMULES_COLUMN_ALIASES = {
        "formule": ["formule", "formules", "nom_formule"],
        "duree": ["duree", "durée"],
        "tarif": ["tarif", "prix", "cout", "coût"],
    }

    NIVEAUX_COLUMN_ALIASES = {
        "niveau": ["niveau", "classe"],
        "annee_de_naissance": ["annee_de_naissance", "annee", "année_de_naissance", "annee_naissance"],
        "cycle_name": ["cycle_name", "cycle", "nom_cycle"],
    }

    @staticmethod
    def _clean_value(value: Any) -> str:
        """
        Nettoie une valeur de cellule.
        """
        if pd.isna(value):
            return ""

        text = str(value).strip()

        if text.lower() in {"", "nan", "none", "null"}:
            return ""

        return text

    @classmethod
    def _clean_row(cls, row: pd.Series, columns: list[str]) -> dict[str, str]:
        """
        Convertit une ligne pandas en dictionnaire nettoyé.
        """
        return {
            col: cls._clean_value(row.get(col, ""))
            for col in columns
        }

    @staticmethod
    def _resolve_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
        """
        Retourne le premier nom de colonne présent dans le DataFrame
        parmi une liste d'alias.
        """
        df_columns = set(df.columns)
        for alias in aliases:
            if alias in df_columns:
                return alias
        return None

    @classmethod
    def _resolve_required_columns(
        cls,
        df: pd.DataFrame,
        alias_mapping: dict[str, list[str]],
        context_name: str,
    ) -> dict[str, str]:
        """
        Résout les colonnes réelles d'un DataFrame à partir d'alias métier.

        Retourne un mapping de type :
        {
            "formule": "formules",
            "duree": "duree",
            "tarif": "tarif"
        }
        """
        resolved: dict[str, str] = {}
        missing: list[str] = []

        for business_field, aliases in alias_mapping.items():
            found_column = cls._resolve_column(df, aliases)
            if found_column is None:
                missing.append(business_field)
            else:
                resolved[business_field] = found_column

        if missing:
            raise ValueError(
                f"Colonnes métier manquantes dans {context_name} : {sorted(missing)}. "
                f"Colonnes disponibles : {sorted(df.columns.tolist())}"
            )

        return resolved

    @staticmethod
    def _deduplicate_records(
        records: list[dict[str, str]],
        unique_keys: list[str],
    ) -> list[dict[str, str]]:
        """
        Supprime les doublons sur la base d'un sous-ensemble de clés.
        """
        seen: set[tuple[str, ...]] = set()
        deduplicated: list[dict[str, str]] = []

        for record in records:
            key = tuple(record.get(field, "") for field in unique_keys)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(record)

        return deduplicated

    @classmethod
    def transform_formules_tarifs(cls, df: pd.DataFrame) -> list[dict[str, str]]:
        """
        Transforme un tableau de formules/tarifs en objets métier.

        Champs métier :
        - categorie
        - formule
        - duree
        - prise_en_charge_pause_dejeuner
        - tarif
        """
        resolved_columns = cls._resolve_required_columns(
            df=df,
            alias_mapping=cls.FORMULES_COLUMN_ALIASES,
            context_name="formules_tarifs",
        )

        results: list[dict[str, str]] = []
        skipped_rows = 0

        for _, row in df.iterrows():
            row_data = cls._clean_row(row, list(df.columns))

            categorie = row_data.get("categorie", "")
            formule = row_data.get(resolved_columns["formule"], "")
            duree = row_data.get(resolved_columns["duree"], "")
            prise_en_charge_pause_dejeuner = row_data.get(
                "prise_en_charge_pause_dejeuner",
                "",
            )
            tarif = row_data.get(resolved_columns["tarif"], "")

            # règle minimale : une formule doit exister
            if not formule:
                skipped_rows += 1
                continue

            results.append(
                {
                    "categorie": categorie,
                    "formule": formule,
                    "duree": duree,
                    "prise_en_charge_pause_dejeuner": prise_en_charge_pause_dejeuner,
                    "tarif": tarif,
                }
            )

        results = cls._deduplicate_records(
            results,
            unique_keys=[
                "categorie",
                "formule",
                "duree",
                "prise_en_charge_pause_dejeuner",
                "tarif",
            ],
        )

        logger.info(
            "BusinessTransformer.formules_tarifs : %s enregistrements générés, %s lignes ignorées",
            len(results),
            skipped_rows,
        )

        return results

    @classmethod
    def transform_niveaux_mapping(cls, df: pd.DataFrame) -> list[dict[str, str]]:
        """
        Transforme un tableau niveaux / année de naissance / cycle
        en mapping métier générique.

        Champs métier attendus :
        - niveau
        - annee_de_naissance
        - cycle_name
        """
        resolved_columns = cls._resolve_required_columns(
            df=df,
            alias_mapping=cls.NIVEAUX_COLUMN_ALIASES,
            context_name="niveaux_mapping",
        )

        results: list[dict[str, str]] = []
        skipped_rows = 0

        for _, row in df.iterrows():
            row_data = cls._clean_row(row, list(df.columns))

            niveau = row_data.get(resolved_columns["niveau"], "")
            annee = row_data.get(resolved_columns["annee_de_naissance"], "")
            cycle_name = row_data.get(resolved_columns["cycle_name"], "")

            # règle minimale : niveau + année
            if not niveau or not annee:
                skipped_rows += 1
                continue

            results.append(
                {
                    "niveau": niveau,
                    "annee": annee,
                    "cycle_name": cycle_name,
                }
            )

        results = cls._deduplicate_records(results, unique_keys=["niveau", "annee", "cycle_name"])

        logger.info(
            "BusinessTransformer.niveaux_mapping : %s enregistrements générés, %s lignes ignorées",
            len(results),
            skipped_rows,
        )
        return results

    @staticmethod
    def build_cycles_niveaux_payload(
        niveaux_mapping: list[dict[str, str]],
        cycles: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Construit le payload métier final des niveaux/cycles.
        """
        payload = {
            "niveaux_par_annee": niveaux_mapping,
            "cycles": cycles or [],
        }

        logger.info(
            "BusinessTransformer.build_cycles_niveaux_payload : payload construit (%s niveaux, %s cycles)",
            len(payload["niveaux_par_annee"]),
            len(payload["cycles"]),
        )
        return payload