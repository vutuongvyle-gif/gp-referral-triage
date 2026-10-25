from langchain_chroma import Chroma

from referral_triage.config import settings
from referral_triage.llm import get_embeddings

_store = None


def store() -> Chroma:
    global _store
    if _store is None:
        _store = Chroma(collection_name="criteria", embedding_function=get_embeddings(),
                        persist_directory=settings.chroma_dir)
    return _store


def retrieve(specialty: str, query: str, k: int = 6) -> list:
    return store().similarity_search(query, k=k, filter={"specialty": specialty})


def format_hits(docs) -> str:
    return "\n\n".join(
        f"[{d.metadata['criterion_id']} | {d.metadata.get('section', '')}]\n{d.page_content}"
        for d in docs
    )