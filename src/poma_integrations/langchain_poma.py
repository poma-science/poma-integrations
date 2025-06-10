"""
POMA helpers for LangChain
──────────────────────────
• Doc2PomaLoader         – BaseLoader  → sentence-aligned markdown
• PomaSentenceSplitter   – TextSplitter → one sentence = one Document
• PomaChunksetSplitter   – TextSplitter → one chunkset = one Document
• PomaCheatsheetRetriever– BaseRetriever→ one cheatsheet = one Document
"""

import os
import pathlib
import zipfile
from typing import Any
from pydantic import PrivateAttr
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader
from langchain_text_splitters import TextSplitter
from langchain.schema.retriever import BaseRetriever
import poma_senter
import doc2poma


__all__ = [
    "Doc2PomaLoader",
    "PomaSentenceSplitter",
    "PomaChunksetSplitter",
    "PomaCheatsheetRetriever",
]


class Doc2PomaLoader(BaseLoader):
    """
    Loads a document by converting it to `.poma` format using the `doc2poma` pipeline,
    and returns it as a single `Langchain Document`.
    """

    def __init__(self, config: dict):
        self.config = config

    def load(self, file_path: str) -> list[Document]:
        archive_path, costs = doc2poma.convert(
            file_path, config=self.config, base_url=None
        )
        print(f"Converted {file_path} to {archive_path} – for USD {costs:.5f}")

        # load markdown, ensure .poma extension
        root, _ = os.path.splitext(archive_path)
        poma_file_path = root + ".poma"
        with zipfile.ZipFile(poma_file_path, "r") as zipf:
            with zipf.open("content.md") as file:
                md = file.read().decode("utf-8")

        return [Document(page_content=md, metadata={"poma_archive": archive_path})]


class PomaSentenceSplitter(TextSplitter):
    """
    Splits input documents into individual sentences using a robust sentence segmenter.
    Each non-empty sentence becomes a new `Document`.

    Note: Only `split_documents()` is implemented. `split_text()` is not supported.
    """

    def split_documents(self, docs: list[Document]) -> list[Document]:
        out = []
        for doc in docs:
            lines = poma_senter.clean_and_segment_text(doc.page_content).splitlines()
            for i, line in enumerate(lines):
                if line.strip():
                    out.append(
                        Document(
                            page_content=line.strip(),
                            metadata={"sentence_idx": i},
                        )
                    )
        return out

    def split_text(self, text: str) -> list[str]:
        raise NotImplementedError("PomaSentenceSplitter supports split_documents()")


class PomaChunksetSplitter(TextSplitter):
    """
    Splits a `.poma` archive into high-level content chunksets using the `poma_chunker` pipeline.
    Each non-empty sentence becomes a new `Document`.

    This splitter expects exactly one input document.

    Returns (doc_id, output_docs, chunks, chunksets) so caller can persist the raw data.
    """

    def __init__(self, config: dict):
        super().__init__(chunk_size=10**9)
        self.config = config

    def split_documents(
        self, docs: list[Document]
    ) -> tuple[str, list[Document], list[dict], list[dict]]:
        from poma_chunker import process

        if len(docs) != 1:
            raise ValueError(f"Expected exactly one document, got {len(docs)}")
        poma_doc = docs[0]
        archive_path = poma_doc.metadata["poma_archive"]
        result = process(archive_path, self.config)
        chunks, chunksets = result["chunks"], result["chunksets"]
        poma_doc_id = pathlib.Path(archive_path).stem
        docs_out = [
            Document(
                page_content=cs["contents"],
                metadata={
                    "doc_id": poma_doc_id,
                    "chunkset_index": cs["chunkset_index"],
                },
            )
            for cs in chunksets
        ]
        return poma_doc_id, docs_out, chunks, chunksets

    def split_text(self, text: str) -> list[str]:
        raise NotImplementedError("PomaChunksetSplitter supports split_documents()")


class PomaCheatsheetRetriever(BaseRetriever):
    """
    Retrieves relevant chunksets based on a given query from a vector store.

    Collapse top-k chunkset data into a single cheatsheet Document (per doc_id).
    """

    _vector_store: Any = PrivateAttr()
    _fetch_chunks: Any = PrivateAttr()
    _fetch_chunkset: Any = PrivateAttr()
    _top_k: int = PrivateAttr()

    def __init__(self, vectorstore, chunks_store, chunkset_store, top_k):
        super().__init__()
        self._vector_store = vectorstore
        self._fetch_chunks = chunks_store
        self._fetch_chunkset = chunkset_store
        self._top_k = top_k

    def _get_relevant_documents(self, query: str) -> list[Document]:
        from poma_chunker import get_relevant_chunks, generate_cheatsheet

        vector_search_results_and_scores = (
            self._vector_store.similarity_search_with_score(query, k=self._top_k)
        )
        if not vector_search_results_and_scores:
            return []

        search_results_per_doc_id = {}
        for search_result, score in vector_search_results_and_scores:
            doc_id = search_result.metadata["doc_id"]
            if doc_id not in search_results_per_doc_id:
                search_results_per_doc_id[doc_id] = []
            search_results_per_doc_id[doc_id].append(search_result)

        result_documents = []
        for doc_id, search_results in search_results_per_doc_id.items():
            doc_chunks = self._fetch_chunks(doc_id)
            doc_chunk_ids = []
            for search_result in search_results:
                chunkset_index = search_result.metadata["chunkset_index"]
                chunkset = self._fetch_chunkset(doc_id, chunkset_index)
                doc_chunk_ids.extend(chunkset["chunks"])

            all_relevant_chunks = get_relevant_chunks(doc_chunk_ids, doc_chunks)
            cheatsheet = generate_cheatsheet(all_relevant_chunks)
            print("── Cheatsheet ──\n", cheatsheet or "(empty)", "\n──────┘\n")

            result_documents.append(
                Document(page_content=cheatsheet, metadata={"source": "poma"})
            )

        return result_documents

    async def _aget_relevant_documents(self, query: str):
        raise NotImplementedError(
            "Async path not implemented for PomaCheatsheetRetriever."
        )
