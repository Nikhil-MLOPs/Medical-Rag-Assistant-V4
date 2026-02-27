import json
import pymupdf
from pathlib import Path

from src.utils.logging import setup_logging
from src.utils.config import load_ingestion_config

# Set up logging for the ingestion process
logger = setup_logging("Ingestion")

FOOTER_KEYWORDS = ["g a l e e n c y c l o p e d i a"] # Footer text to remove from each page

# Helper function to clean footer text from each page
def clean_footer(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lower = line.lower()
        if any(k in lower for k in FOOTER_KEYWORDS):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def ingest():
    # Ingestion of configuration parameters
    cfg = load_ingestion_config("configs/ingestion.yaml")

    raw_dir = Path(cfg.raw_dir)
    out_dir = Path(cfg.processed_dir) # Gives data/processed/pages as value from the config file since processed_dir is key.
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "pages.jsonl"

    total_pages = 0

    with open(out_file, "w", encoding="utf-8") as f_out:

        for pdf_path in raw_dir.glob("*.pdf"):
            logger.info(f"Ingesting {pdf_path.stem} with extension {pdf_path.suffix} and size {pdf_path.stat().st_size / (1024 * 1024):.2f} MB")
            doc = pymupdf.open(pdf_path)

            doc_page_count = doc.page_count
            logger.info(f"Total pages in the document {pdf_path.stem} are {doc_page_count}.")

            for page_index in range(doc_page_count):
                if page_index < cfg.skip_start_pages:
                    continue
                if page_index > cfg.skip_after_pages:
                    break

                page = doc.load_page(page_index)
                text = page.get_text("text")
                text = clean_footer(text)

                if not text.strip():
                    continue

                record = {"text": text, "metadata": {"pdf": pdf_path.stem, "page": page_index + 1}}

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_pages += 1

    logger.info(f"Total pages ingested: {total_pages}")
    logger.info(f"Wrote pages to {out_file}")


if __name__ == "__main__":
    ingest()