"""Chunk criteria Markdown by criterion and store embeddings in Chroma."""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter

from referral_triage.config import settings
from referral_triage.llm import get_embeddings

SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("##", "criterion"), ("###", "section")], strip_headers=False
)


def load_docs():
    docs = []
    for path in sorted(Path("data/criteria").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        _, front, body = text.split("---", 2)           # simple front-matter split
        meta = dict(line.split(": ", 1) for line in front.strip().splitlines())
        for d in SPLITTER.split_text(body):
            crit = d.metadata.get("criterion", "")
            d.metadata.update(
                specialty=meta["specialty"],
                criterion_id=crit.split(" ")[0] if crit else "",
                source=path.name,
            )
            docs.append(d)
    return docs


def main():
    store = Chroma(collection_name="criteria", embedding_function=get_embeddings(),
                   persist_directory=settings.chroma_dir)
    store.reset_collection()                            # rebuild from scratch
    docs = load_docs()
    store.add_documents(docs)
    print(f"Indexed {len(docs)} chunks")


if __name__ == "__main__":
    main()