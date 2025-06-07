# 🍍 POMA ↔ LangChain & Llama-Index mini-SDK

[![PyPI version](https://img.shields.io/pypi/v/poma-integrations.svg)](https://pypi.org/project/poma-integrations/)  
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](LICENSE)

> **Part of the POMA toolkit.** For the complete documentation, see the [organization README](https://github.com/poma-science/.github).

**POMA-Integrations** lets you turn any PDF, HTML, or image into:
- Clean markdown
- Structure-preserving chunksets
- Token-efficient cheatsheets

…with native extension points for LangChain and LlamaIndex. Just plug in your own storage and retrieval logic.

| Layer               | LangChain class                             | Llama-Index class                       |
|---------------------|---------------------------------------------|-----------------------------------------|
| **Loader / Reader** | `Doc2PomaLoader`                            | `Doc2PomaReader`                        |
| **Sentence splitter** | `PomaSentenceSplitter`                    | `PomaSentenceNodeParser`                |
| **Chunkset splitter** | `PomaChunksetSplitter`                    | `PomaChunksetNodeParser`                |
| **Cheatsheet builder** | `PomaCheatsheetRetriever`                | `PomaCheatsheetPostProcessor`           |

**You supply:**
- A `chunk_store` callback (to persist chunksets)
- A `chunk_fetcher` callback (to retrieve raw sentences by `(doc_id, chunk_ids)`)

Store your data in SQLite, Postgres, Redis, S3—whatever fits your stack.

---

> ⚡ **Why POMA chunking?**
>
> Most RAG tools split documents linearly, losing structure and context. POMA preserves the document tree—so headings, lists, and sections never lose their meaning. This enables context-rich retrieval, minimizes hallucinations, and saves tokens. See the [organization README](https://github.com/poma-science/.github) for real-world examples.

---

## See Also

| Module | Description | PyPI | GitHub |
|--------|-------------|------|--------|
| **doc2poma** | Convert any file (PDF, HTML, image, Markdown) to .poma | [PyPI](https://pypi.org/project/doc2poma/) | [GitHub](https://github.com/poma-science/doc2poma) |
| **poma-senter** | Clean & split into one sentence per line | [PyPI](https://pypi.org/project/poma-senter/) | [GitHub](https://github.com/poma-science/poma-senter) |
| **poma-chunker** | Build depth-aware chunks & chunksets | [PyPI](https://pypi.org/project/poma-chunker/) | [GitHub](https://github.com/poma-science/poma-chunker) |
| **poma-integrations** | Drop-in classes for LangChain / LlamaIndex | [PyPI](https://pypi.org/project/poma-integrations/) | [GitHub](https://github.com/poma-science/poma-integrations) |

---

## Quick Start

1. **Convert** your document using `Doc2PomaLoader` (or `Doc2PomaReader` for LlamaIndex).
2. **Split** the document into chunksets with `PomaChunksetSplitter`.
3. **Store** chunksets and raw chunks using your own `chunk_store` callback.
4. **Retrieve** chunks with your `chunk_fetcher` callback for downstream use (retrieval, QA, etc).

**Pseudocode:**
```python
from poma_integrations.langchain_poma import Doc2PomaLoader, PomaChunksetSplitter

def my_chunk_store(doc_id, chunks):
    # Store chunks in your DB
    ...

def my_chunk_fetcher(doc_id, chunk_ids):
    # Retrieve chunks from your DB
    ...

loader = Doc2PomaLoader(chunk_store=my_chunk_store, chunk_fetcher=my_chunk_fetcher)
docs = loader.load(["/path/to/my.pdf"])
chunks = PomaChunksetSplitter().split_documents(docs)
```

---

## LangChain Integration – Minimal Example & Workflow

```python
from poma_integrations.langchain_poma import Doc2PomaLoader, PomaChunksetSplitter, PomaCheatsheetRetriever

# 1. Convert and chunk your document
loader = Doc2PomaLoader(chunk_store=my_chunk_store, chunk_fetcher=my_chunk_fetcher)
docs = loader.load(["/path/to/my.pdf"])
chunks = PomaChunksetSplitter().split_documents(docs)

# 2. Build a retriever and chain (see example.py for full implementation)
retriever = PomaCheatsheetRetriever(vectorstore, my_chunk_fetcher)
```

**Typical workflow:**
1. Load and convert documents with `Doc2PomaLoader`.
2. Split into chunksets with `PomaChunksetSplitter`.
3. Store and fetch chunks using your callbacks.
4. Create a retriever (e.g. `PomaCheatsheetRetriever`) and plug into your favorite QA or RAG pipeline.

➡️ _See [`src/poma_integrations/langchain_poma.py`](./src/poma_integrations/langchain_poma.py) for the full LangChain integration code_

---

## LlamaIndex Integration – Minimal Example & Workflow

```python
from poma_integrations.llamaindex_poma import Doc2PomaReader, PomaChunksetNodeParser, PomaCheatsheetPostProcessor

# 1. Convert and chunk your document
reader = Doc2PomaReader(chunk_store=my_chunk_store, chunk_fetcher=my_chunk_fetcher)
nodes = reader.load_data("/path/to/my.pdf")
chunksets, raw_chunks = PomaChunksetNodeParser().get_nodes_from_documents(nodes)

# 2. Build a post-processor and query engine (see example_llamaindex.py for full implementation)
post = PomaCheatsheetPostProcessor(chunk_fetcher=my_chunk_fetcher)
```

**Typical workflow:**
1. Load and convert documents with `Doc2PomaReader`.
2. Split into chunksets with `PomaChunksetNodeParser`.
3. Store and fetch chunks using your callbacks.
4. Use `PomaCheatsheetPostProcessor` in your LlamaIndex query engine.

➡️ _See [`src/poma_integrations/llamaindex_poma.py`](./src/poma_integrations/llamaindex_poma.py) for the full LlamaIndex integration code_

---

## Main Classes

| Class                        | Description                                 |
|------------------------------|---------------------------------------------|
| `Doc2PomaLoader`             | Loads and converts documents (LangChain)    |
| `PomaSentenceSplitter`       | Splits documents into sentences             |
| `PomaChunksetSplitter`       | Splits sentences into chunksets             |
| `PomaCheatsheetRetriever`    | Retrieval pipeline for cheatsheets (LC)     |
| `Doc2PomaReader`             | Loads and converts documents (LlamaIndex)   |
| `PomaSentenceNodeParser`     | Splits into sentences (LlamaIndex)          |
| `PomaChunksetNodeParser`     | Splits into chunksets (LlamaIndex)          |
| `PomaCheatsheetPostProcessor`| Post-processes results for cheatsheets      |

---

## 🚀 Installation

**From local repository**
```bash
pip install .
```

**From everywhere with pypi**
```bash
pip install poma-integrations
```

*Optionally add `langchain` or `llama-index-core` to the same command.*

---

Enjoy hierarchy-aware RAG with **one import** instead of hundreds of lines of
glue code.
