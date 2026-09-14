from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging_config import setup_logging

from batch.ingestion.sheet_loader import SheetLoader
from batch.processing.business_transformer import BusinessTransformer
from batch.processing.cycle_parser import CycleParser


logger = logging.getLogger(__name__)


class BusinessBatchRunner:
    """
    Orchestration du pipeline batch pour les données métier structurées.

    Sorties principales :
    - formules_tarifs.json
    - cycles_niveaux.json
    """

    def __init__(
        self,
        raw_sheets_dir: Path,
        raw_docs_dir: Path,
        business_dir: Path,
    ) -> None:
        self.raw_sheets_dir = Path(raw_sheets_dir)
        self.raw_docs_dir = Path(raw_docs_dir)
        self.business_dir = Path(business_dir)

    def run(self) -> dict[str, Any]:
        """
        Exécute l'ensemble du pipeline business.
        """
        logger.info("Démarrage du pipeline business")

        formules_tarifs = self.build_formules_tarifs()
        cycles_niveaux = self.build_cycles_niveaux()

        stats = {
            "formules_tarifs_count": len(formules_tarifs),
            "niveaux_mapping_count": len(cycles_niveaux.get("niveaux_par_annee", [])),
            "cycles_count": len(cycles_niveaux.get("cycles", [])),
        }

        logger.info(
            "Pipeline business terminé : %s tarifs, %s mappings niveaux, %s cycles",
            stats["formules_tarifs_count"],
            stats["niveaux_mapping_count"],
            stats["cycles_count"],
        )

        return {
            "formules_tarifs": formules_tarifs,
            "cycles_niveaux": cycles_niveaux,
            "stats": stats,
        }

    def build_formules_tarifs(self) -> list[dict[str, str]]:
        """
        Génère formules_tarifs.json à partir de formules_tarifs.xlsx
        """
        input_path = self.raw_sheets_dir / "formules_tarifs.xlsx"
        output_path = self.business_dir / "formules_tarifs.json"

        df = self._load_sheet(input_path)
        tarifs = BusinessTransformer.transform_formules_tarifs(df)

        self._save_json(tarifs, output_path)

        logger.info(
            "Fichier formules_tarifs généré : %s enregistrements",
            len(tarifs),
        )
        return tarifs

    def build_cycles_niveaux(self) -> dict[str, Any]:
        """
        Génère cycles_niveaux.json à partir de :
        -niveaux_arabe.xlsx
        - niveaux_arabe.docx
        """
        niveaux_sheet_path = self.raw_sheets_dir / "niveaux_arabe.xlsx"
        niveaux_doc_path = self.raw_docs_dir / "niveaux_arabe.docx"
        output_path = self.business_dir / "cycles_niveaux.json"

        df = self._load_sheet(niveaux_sheet_path)
        niveaux_mapping = BusinessTransformer.transform_niveaux_mapping(df)

        cycles = CycleParser.parse_cycles(niveaux_doc_path)

        payload = BusinessTransformer.build_cycles_niveaux_payload(
            niveaux_mapping=niveaux_mapping,
            cycles=cycles,
        )

        self._save_json(payload, output_path)

        logger.info(
            "Fichier cycles_niveaux généré : %s niveaux, %s cycles",
            len(payload.get("niveaux_par_annee", [])),
            len(payload.get("cycles", [])),
        )
        return payload

    @staticmethod
    def _load_sheet(file_path: Path):
        """
        Charge et nettoie un fichier tabulaire avec le SheetLoader.
        """
        loader = SheetLoader(
            file_path=file_path,
            normalize_columns=True,
            drop_empty_rows=True,
            strip_string_values=True,
        )
        return loader.load()

    @staticmethod
    def _save_json(data: Any, output_path: Path) -> None:
        """
        Sauvegarde un objet Python en JSON UTF-8.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Fichier JSON sauvegardé : %s", output_path)


def main() -> dict[str, Any]:
    setup_logging("INFO")
    settings = get_settings()

    runner = BusinessBatchRunner(
        raw_sheets_dir=settings.raw_data_path / "google_sheets",
        raw_docs_dir=settings.raw_data_path / "google_drive_docs",
        business_dir=settings.business_data_path,
    )

    return runner.run()


if __name__ == "__main__":
    main()