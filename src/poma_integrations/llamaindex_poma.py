"""
POMA helpers for Llama-Index
────────────────────────────
• Doc2PomaReader          – BaseReader
• PomaSentenceNodeParser  – NodeParser
• PomaChunksetNodeParser  – NodeParser
• PomaCheatsheetPostProcessor – BaseNodePostprocessor
"""
from __future__ import annotations
import json, typing as t, pathlib
import zipfile
from llama_index.core.schema import NodeWithScore, Document as LIDoc, TextNode
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.postprocessor.types import BaseNodePostprocessor
import doc2poma, poma_chunker
from poma_senter import clean_and_segment_text
from pydantic import PrivateAttr

__all__ = [
    "Doc2PomaReader",
    "PomaSentenceNodeParser",
    "PomaChunksetNodeParser",
    "PomaCheatsheetPostProcessor",
]

# ---------------------------------------------------------------------- #
#  READER                                                                #
# ---------------------------------------------------------------------- #
class Doc2PomaReader(BaseReader):
    """Outputs one Llama-Index Document with sentence-aligned markdown."""
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def load_data(self, file_path: str) -> list[LIDoc]:
        archive_result = doc2poma.convert(file_path, base_url=None, config=self.cfg)
        
        # Handle both string and tuple return values for backward compatibility
        if isinstance(archive_result, tuple):
            archive_path, cost = archive_result
        else:
            archive_path = archive_result
            cost = 0
            
        print(f"Converted {file_path} to {archive_path} (cost: ${cost})")

        archive_path = pathlib.Path(archive_path).with_suffix(".poma")
        
        # Convert PosixPath to string if needed
        if isinstance(archive_path, pathlib.Path):
            archive_path_str = str(archive_path)
        else:
            archive_path_str = archive_path

        with zipfile.ZipFile(archive_path_str, "r") as zipf:
            with zipf.open("content.md") as f:
                md = f.read().decode("utf-8")
        return [LIDoc(text=md, metadata={"poma_archive": archive_path})]


# ---------------------------------------------------------------------- #
#  NODE PARSERS                                                          #
# ---------------------------------------------------------------------- #
class PomaSentenceNodeParser(SimpleNodeParser):
    """One TextNode per sentence (robust segmentation)."""
    def get_nodes_from_documents(self, docs: list[LIDoc], **_) -> list[TextNode]:
        out = []
        for d in docs:
            lines = clean_and_segment_text(d.text).splitlines()
            for i, line in enumerate(lines):
                if line.strip():
                    out.append(TextNode(text=line.strip(), metadata={"sentence_idx": i}))
        return out


class PomaChunksetNodeParser(SimpleNodeParser):
    """One TextNode per chunkset. Returns (nodes, raw_chunks)"""

    _cfg: dict = PrivateAttr()

    def __init__(self, cfg: dict):
        super().__init__()
        self._cfg = cfg

    def get_nodes_from_documents(self, docs: list[LIDoc], **_) -> tuple[list[TextNode], list[dict]]:
        if len(docs) != 1:
            raise ValueError("Pass the markdown Document only")
        md_doc = docs[0]
        archive = md_doc.metadata["poma_archive"]
        
        # Handle both string and tuple return values for backward compatibility
        if isinstance(archive, tuple):
            archive_path, _ = archive
        else:
            archive_path = archive
        
        # Convert PosixPath to string if needed
        if isinstance(archive_path, pathlib.Path):
            archive_path = str(archive_path)
            
        res = poma_chunker.process(archive_path, self._cfg)
        doc_id = pathlib.Path(archive_path).stem

        nodes = [
            TextNode(
                text=cs["contents"],
                metadata={
                    "doc_id": doc_id,
                    "chunk_ids": json.dumps(cs["chunks"]),
                    "chunkset_id": cs["chunkset_index"],
                },
            )
            for cs in res["chunksets"]
        ]
        return nodes, res["chunks"]


# ---------------------------------------------------------------------- #
#  POST-PROCESSOR                                                        #
# ---------------------------------------------------------------------- #
ChunkFetcher = t.Callable[[str, t.Sequence[int]], t.List[dict]]

class PomaCheatsheetPostProcessor(BaseNodePostprocessor):
    """
    Collapse top-k chunkset nodes into a single cheatsheet node.

    Needs a `chunk_fetcher` callable (same signature as LangChain retriever).
    """
    _fetch: ChunkFetcher = PrivateAttr()

    def __init__(self, chunk_fetcher: ChunkFetcher):
        super().__init__()
        self._fetch = chunk_fetcher


    def _postprocess_nodes(
        self,
        nodes,
        query_bundle=None,
        **_,
    ) -> list[TextNode]:
        if not nodes:
            return []

        if query_bundle:
            print("query_bundle is not used in PomaCheatsheetPostProcessor")

        # Group chunk_ids and max scores by doc_id
        doc_id_to_chunk_ids = {}
        doc_id_to_max_score = {}
        for n in nodes:
            doc_id = n.metadata["doc_id"]
            chunk_ids = json.loads(n.metadata["chunk_ids"])
            score = getattr(n, "score", 1.0)  # default to 1.0 if score is missing

            if doc_id not in doc_id_to_chunk_ids:
                doc_id_to_chunk_ids[doc_id] = []
                doc_id_to_max_score[doc_id] = score
            else:
                doc_id_to_max_score[doc_id] = max(doc_id_to_max_score[doc_id], score)

            doc_id_to_chunk_ids[doc_id].extend(chunk_ids)

        result_nodes = []

        # Process each group of nodes by doc_id
        for doc_id, chunk_ids in doc_id_to_chunk_ids.items():
            raw = self._fetch(doc_id)
            enriched = poma_chunker.get_relevant_chunks(chunk_ids, raw)
            cheat = poma_chunker.generate_cheatsheet(enriched)
            cheat_node = TextNode(text=cheat, metadata={"source": "poma-cheatsheet"})
            score = doc_id_to_max_score[doc_id]
            result_nodes.append(NodeWithScore(node=cheat_node, score=score))

        return result_nodes
