"""
POMA helpers for LangChain
──────────────────────────
• Doc2PomaLoader         – BaseLoader  → sentence-aligned markdown
• PomaSentenceSplitter   – TextSplitter → one sentence = one Document
• PomaChunksetSplitter   – TextSplitter → one chunkset  = one Document
• PomaCheatsheetRetriever– BaseRetriever→ emits ONE cheatsheet Document
"""
from __future__ import annotations
import json, typing as t, pathlib, sqlite3
import zipfile
from pydantic import PrivateAttr
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TextSplitter
from langchain.schema.retriever import BaseRetriever
import doc2poma, poma_chunker
from poma_senter import clean_and_segment_text

__all__ = [
    "Doc2PomaLoader",
    "PomaSentenceSplitter",
    "PomaChunksetSplitter",
    "PomaCheatsheetRetriever",
]

# ---------------------------------------------------------------------- #
#  LOADER                                                                #
# ---------------------------------------------------------------------- #
class Doc2PomaLoader(BaseLoader):
    """Convert any file to markdown – one sentence per line."""
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def load(self, file_path: str) -> list[Document]:
        archive_result = doc2poma.convert(file_path, base_url=None, config=self.cfg)
        
        # Handle both string and tuple return values for backward compatibility
        if isinstance(archive_result, tuple):
            archive_path, cost = archive_result
        else:
            archive_path = archive_result
            cost = 0
            
        print(f"Converted {file_path} to {archive_path} (cost: ${cost})")

        archive_path = pathlib.Path(archive_path).with_suffix(".poma")

        with zipfile.ZipFile(archive_path, "r") as zipf:
            with zipf.open("content.md") as f:
                md = f.read().decode("utf-8")
        return [Document(page_content=md, metadata={"poma_archive": archive_path})]


# ---------------------------------------------------------------------- #
#  SPLITTERS                                                             #
# ---------------------------------------------------------------------- #
class PomaSentenceSplitter(TextSplitter):
    """Robust sentence segmentation (one Document per sentence)."""
    def split_documents(self, docs: list[Document]) -> list[Document]:
        out = []
        for doc in docs:
            lines = clean_and_segment_text(doc.page_content).splitlines()
            for i, line in enumerate(lines):
                if line.strip():
                    out.append(
                        Document(
                            page_content=line.strip(),
                            metadata={**doc.metadata, "sentence_idx": i},
                        )
                    )
        return out

    def split_text(self, text: str) -> list[str]:
        raise NotImplementedError("PomaSentenceSplitter supports split_documents()")

class PomaChunksetSplitter(TextSplitter):
    """
    Structural splitter – one Document per POMA chunkset.

    Returns (docs, raw_chunks) so caller can persist raw chunks.
    """
    def __init__(self, cfg: dict):
        super().__init__(chunk_size=10**9)
        self.cfg = cfg

    def split_documents(self, docs: list[Document]) -> tuple[list[Document], list[dict]]:
        if len(docs) != 1:
            raise ValueError("Pass a single Document (the markdown) to ChunksetSplitter")
        md_doc = docs[0]
        archive = md_doc.metadata["poma_archive"]
        
        # Convert PosixPath to string if needed
        if isinstance(archive, pathlib.Path):
            archive_path = str(archive)
        else:
            archive_path = archive
            
        res = poma_chunker.process(archive_path, self.cfg)
        doc_id = pathlib.Path(archive_path).stem

        docs_out = [
            Document(
                page_content=cs["contents"],
                metadata={
                    "doc_id": doc_id,
                    "chunk_ids": json.dumps(cs["chunks"]),
                    "chunkset_id": cs["chunkset_index"],
                },
            )
            for cs in res["chunksets"]
        ]
        return docs_out, res["chunks"]  # second element for persistence
    
    def split_text(self, text: str) -> list[str]:
        raise NotImplementedError("PomaChunksetSplitter supports split_documents()")


# ---------------------------------------------------------------------- #
#  RETRIEVER                                                             #
# ---------------------------------------------------------------------- #
ChunkFetcher = t.Callable[[str, t.Sequence[int]], t.List[dict]]

class PomaCheatsheetRetriever(BaseRetriever):
    """
    Wrap any VectorStore. Needs a callback that returns raw chunk dicts.

        fetch(doc_id, list_of_ids) -> list[{chunk_index, depth, content}]
    """
    _vs: t.Any = PrivateAttr()
    _fetch: t.Any = PrivateAttr()
    _k: int = PrivateAttr(default=4)

    def __init__(self, vectorstore, chunks_store, k=4):
        super().__init__()
        self._vs = vectorstore
        self._fetch = chunks_store
        self._k = k


    def _get_relevant_documents(self, query: str):
        hits = self._vs.similarity_search(query, k=self._k)
        if not hits:
            return []

        # Group hits by doc_id using a regular dictionary
        doc_id_to_chunk_ids = {}
        for h in hits:
            doc_id = h.metadata["doc_id"]
            chunk_ids = json.loads(h.metadata["chunk_ids"])
            if doc_id not in doc_id_to_chunk_ids:
                doc_id_to_chunk_ids[doc_id] = []
            doc_id_to_chunk_ids[doc_id].extend(chunk_ids)

        result_documents = []

        # Process each group of hits by doc_id
        for doc_id, chunk_ids in doc_id_to_chunk_ids.items():
            raw_chunks = self._fetch(doc_id)
            enriched = poma_chunker.get_relevant_chunks(chunk_ids, raw_chunks)
            cheat = poma_chunker.generate_cheatsheet(enriched)
            result_documents.append(Document(page_content=cheat, metadata={"source": "poma"}))

        return result_documents



    async def _aget_relevant_documents(self, query: str):
        raise NotImplementedError("Async path not implemented for PomaCheatsheetRetriever.")