import json
from pathlib import Path
from typing import List, Dict, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logging import setup_logging
from src.utils.config import load_cleaning_config, load_ingestion_config, CleaningConfig

# Setup Logging for cleaning stage
logger = setup_logging("Cleaning")

# Configuration paths
CONFIG_INGESTION_PATH = Path("configs/ingestion.yaml")
CONFIG_CLEANING_PATH = Path("configs/cleaning.yaml")

# Used to detect structure inside messy PDF text.
SECTION_HEADERS = {
    "definition",
    "description",
    "purpose",
    "preparation",
    "causes and symptoms",
    "causes",
    "symptoms",
    "diagnosis",
    "treatment",
    "alternative treatment",
    "alternative treatments",
    "prevention",
    "prognosis",
    "risks",
    "aftercare",
    "normal results",
    "abnormal results",
    "precautions",
    "cost",
    "results",
    "key terms",
}

# Characters to remove from PDF text.
CONTROL_CHARS = ["\u0002"]

# Removes control characters and trims whitespace.
def clean_line(line: str) -> str:
    for ch in CONTROL_CHARS:
        line = line.replace(ch, "")
    return line.strip()

# Checks if a line is irrelevant or junk. Returns True if: the line is empty (not line), consists only of digits or is very short (≤2 characters)
def is_noise_line(line: str) -> bool:
    return not line or line.isdigit() or len(line) <= 2

# Exclude navigational text (see)that's not core content
def is_cross_reference(line: str) -> bool:
    return " see " in line.lower()

# Lowercase, strip whitespace, remove trailing colon ("Definition:" becomes "definition") strip again. Checks if this string is in the SECTION_HEADERS set.
def is_section_header(line: str) -> bool:
    normalized = line.lower().strip().rstrip(":").strip()
    return normalized in SECTION_HEADERS

# If there are 2-4 words, checks if every non-empty word starts with an uppercase letter (e.g., "John Doe" or "M.D. Ph.D.").Used to detect author credits or bylines, which might be noise in the main content.
def is_author_line(line: str) -> bool:
    words = line.split()
    if 2 <= len(words) <= 4:
        return all(w[0].isupper() for w in words if w)
    return False

# Checks for single uppercase letters ("A" or "B" ypto "Z"), the alphabetical section headers in text.
def is_alphabet_header(line: str) -> bool:
    return len(line) == 1 and line.isalpha() and line.isupper()

# Takes a list of lines and merges those split by hyphens.
def merge_hyphenated_lines(lines: List[str]) -> List[str]:
    merged: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (line.endswith("-") and i + 1 < len(lines) and lines[i + 1] and lines[i + 1][0].islower()):
            merged.append(line[:-1] + lines[i + 1])
            i += 2
        else:
            merged.append(line)
            i += 1
    return merged

# Detects the start of a new topic (e.g. "Diabetes")
def detect_topic(lines: List[str], idx: int) -> Tuple[str | None, int]:
    def is_valid_topic_line(line: str) -> bool:
        return not (is_cross_reference(line) or is_author_line(line) or is_section_header(line) or ";" in line)

    line = lines[idx]

    if (idx + 1 < len(lines) and lines[idx + 1].lower() == "definition" and is_valid_topic_line(line)):
        return line.strip(), 1

    if (idx + 2 < len(lines) and lines[idx + 2].lower() == "definition" and is_valid_topic_line(line) and is_valid_topic_line(lines[idx + 1])):
        return f"{line} {lines[idx + 1]}".strip(), 2

    return None, 0

# Chunk Emission
def _emit_chunks(chunks: List[Document], buffers: Dict[str, List[str]], topic: str | None, page_meta: dict, splitter: RecursiveCharacterTextSplitter):
    if not topic:
        return

    for section, lines in buffers.items():
        if not lines:
            continue

        text = " ".join(lines).replace("  ", " ").strip()

        split_texts = splitter.split_text(text)

        for chunk in split_texts:
            chunks.append(Document(page_content=chunk, metadata={"topic": topic, "section": section, "pdf": page_meta.get("pdf"), "page": page_meta.get("page")}))

# Core Cleaning + Chunking Logic
def clean_and_chunk(pages: List[Document], cfg: CleaningConfig) -> List[Document]:
    logger.info(f"Initializing splitter | chunk_size={cfg.chunk_size} | " f"overlap={cfg.chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    chunks: List[Document] = []
    current_topic = None
    current_section = None
    section_buffers: Dict[str, List[str]] = {}

    for page in pages:
        pdf_name = page.metadata.get("pdf", "unknown")
        page_num = page.metadata.get("page", "unknown")

        raw_lines = [
            clean_line(l)
            for l in page.page_content.splitlines()
            if clean_line(l)
        ]

        raw_lines = merge_hyphenated_lines(raw_lines)

        i = 0

        while i < len(raw_lines):
            line = raw_lines[i]

            if is_noise_line(line) or is_alphabet_header(line):
                i += 1
                continue

            topic, consumed = detect_topic(raw_lines, i)
            if topic:
                if current_topic:
                    _emit_chunks(chunks, section_buffers, current_topic, page.metadata, splitter)

                logger.info(f"[{pdf_name} p.{page_num}] New Topic: {topic}")

                current_topic = topic
                current_section = "definition"
                section_buffers = {"definition": []}
                i += consumed + 1
                continue

            if is_section_header(line):
                current_section = line.lower().strip().rstrip(":").strip()
                section_buffers.setdefault(current_section, [])
                i += 1
                continue

            if current_section and current_topic:
                section_buffers[current_section].append(line)

            i += 1

        # Emit page content
        _emit_chunks(chunks, section_buffers, current_topic, page.metadata, splitter)

        section_buffers = {k: [] for k in section_buffers}

    logger.info(f"Cleaning complete. Generated {len(chunks)} chunks.")
    return chunks

# Main Execution (Pipeline Stage Entry Point)
def main():
    logger.info("Starting Cleaning Stage")

    ing_cfg = load_ingestion_config(CONFIG_INGESTION_PATH)
    clean_cfg = load_cleaning_config(CONFIG_CLEANING_PATH)

    pages_file = Path(ing_cfg.processed_dir) / "pages.jsonl"
    output_dir = Path("data/processed/chunks")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pages_file.exists():
        raise FileNotFoundError(f"Pages file not found: {pages_file}")

    pages: List[Document] = []

    with open(pages_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pages.append(
                Document(page_content=record["text"], metadata=record["metadata"]))

    chunks = clean_and_chunk(pages, clean_cfg)

    output_file = output_dir / "chunks.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({"text": chunk.page_content, "metadata": chunk.metadata}, ensure_ascii=False)+ "\n")

    logger.info(f"Saved {len(chunks)} chunks to {output_file}")


if __name__ == "__main__":
    main()