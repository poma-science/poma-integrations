"""
POMA helpers for Llama-Index (FIXED)
────────────────────────────
• Doc2PomaReader                – BaseReader
• PomaSentenceNodeParser        – NodeParser
• PomaChunksetNodeParser        – NodeParser
• PomaCheatsheetPostProcessor   – BaseNodePostprocessor
"""

import os
import pathlib
import zipfile
from typing import Callable
from pydantic import PrivateAttr
from llama_index.core.schema import NodeWithScore, Document as LIDoc, TextNode
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.postprocessor.types import BaseNodePostprocessor
import poma_senter
import doc2poma


__all__ = [
    "Doc2PomaReader",
    "PomaSentenceNodeParser",
    "PomaChunksetNodeParser",
    "PomaCheatsheetPostProcessor",
]


class Doc2PomaReader(BaseReader):
    """
    Loads a document by converting it to `.poma` format using the `doc2poma` pipeline,
    and returns it as a single `Llama-Index Document`.
    """

    def __init__(self, config: dict):
        self.config = config

    def load_data(self, file_path: str) -> list[LIDoc]:
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

        return [LIDoc(text=md, metadata={"poma_archive": archive_path})]


class PomaSentenceNodeParser(SimpleNodeParser):
    """
    Splits input documents into individual sentences using a robust sentence segmenter.
    Each non-empty sentence becomes a `TextNode`.
    """

    def get_nodes_from_documents(self, docs: list[LIDoc], **_) -> list[TextNode]:
        out = []
        for doc in docs:
            lines = poma_senter.clean_and_segment_text(doc.text).splitlines()
            for i, line in enumerate(lines):
                if line.strip():
                    out.append(
                        TextNode(
                            text=line.strip(),
                            metadata={"sentence_idx": i},
                        )
                    )
        return out


class PomaChunksetNodeParser(SimpleNodeParser):
    """
    Parses a `.poma` archive into high-level content chunksets using the `poma_chunker` pipeline.
    Each chunkset becomes a `TextNode`.

    This parser expects exactly one input document.

    Returns (doc_id, nodes, chunks, chunksets) so caller can persist the raw data.
    """

    _config: dict = PrivateAttr()

    def __init__(self, config: dict):
        super().__init__()
        self._config = config

    def get_nodes_from_documents(
        self, docs: list[LIDoc], **_
    ) -> tuple[str, list[TextNode], list[dict], list[dict]]:
        from poma_chunker import process

        if len(docs) != 1:
            raise ValueError(f"Expected exactly one document, got {len(docs)}")
        poma_doc = docs[0]
        archive_path = poma_doc.metadata["poma_archive"]
        result = process(archive_path, self._config)
        chunks, chunksets = result["chunks"], result["chunksets"]
        poma_doc_id = pathlib.Path(archive_path).stem
        nodes = [
            TextNode(
                text=cs["contents"],
                metadata={
                    "doc_id": poma_doc_id,
                    "chunkset_index": cs["chunkset_index"],
                },
            )
            for cs in chunksets
        ]
        return poma_doc_id, nodes, chunks, chunksets


ChunksFetcher = Callable[[str], list[dict]]
ChunksetFetcher = Callable[[str, int], dict]


class PomaCheatsheetPostProcessor(BaseNodePostprocessor):
    """
    A post-processor that generates cheatsheets from a collection of nodes
    based on relevant chunksets of data.

    Collapses top-k chunkset nodes into a single cheatsheet node (per doc_id).
    """

    _fetch_chunks: ChunksFetcher = PrivateAttr()
    _fetch_chunkset: ChunksetFetcher = PrivateAttr()

    def __init__(self, chunk_fetcher: ChunksFetcher, chunkset_fetcher: ChunksetFetcher):
        super().__init__()
        self._fetch_chunks = chunk_fetcher
        self._fetch_chunkset = chunkset_fetcher

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle=None,
        **kwargs,
    ) -> list[NodeWithScore]:
        from poma_chunker import get_relevant_chunks, generate_cheatsheet

        if not nodes:
            return []

        nodes_per_doc_id = {}
        scores_per_doc_id = {}
        for node in nodes:
            doc_id = node.metadata["doc_id"]
            if doc_id not in nodes_per_doc_id:
                nodes_per_doc_id[doc_id] = []
                scores_per_doc_id[doc_id] = []
            nodes_per_doc_id[doc_id].append(node)
            scores_per_doc_id[doc_id].append(
                node.score if node.score is not None else 1.0
            )

        result_nodes = []
        for doc_id, doc_nodes in nodes_per_doc_id.items():
            doc_chunks = self._fetch_chunks(doc_id)
            doc_chunk_ids = []
            for node in doc_nodes:
                chunkset_index = node.metadata["chunkset_index"]
                chunkset = self._fetch_chunkset(doc_id, chunkset_index)
                doc_chunk_ids.extend(chunkset["chunks"])

            all_relevant_chunks = get_relevant_chunks(doc_chunk_ids, doc_chunks)
            cheatsheet = generate_cheatsheet(all_relevant_chunks)
            print("── Cheatsheet ──\n", cheatsheet or "(empty)", "\n──────┘\n")

            max_score = max(scores_per_doc_id[doc_id])
            cheat_node = TextNode(
                text=cheatsheet, metadata={"source": "poma", "doc_id": doc_id}
            )
            result_nodes.append(NodeWithScore(node=cheat_node, score=max_score))

        return result_nodes
