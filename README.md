# 🍍 POMA ↔ LangChain & Llama-Index mini-SDK

[![PyPI version](https://img.shields.io/pypi/v/poma-integrations.svg)](https://pypi.org/project/poma-integrations/)  
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](LICENSE)

> **Part of the POMA toolkit.** For the complete documentation, see the [organization README](https://github.com/poma-science/.github).

**POMA-Integrations** lets you turn any PDF, HTML, or image into:
- Clean markdown
- Structure-preserving chunksets
- Token-efficient cheatsheets

…with native extension points for LangChain and LlamaIndex. Just plug in your own storage and retrieval logic–SQLite, Postgres, Redis, S3—whatever fits your stack.

| Layer               | LangChain class                             | Llama-Index class                       |
|---------------------|---------------------------------------------|-----------------------------------------|
| **Loader / Reader** | `Doc2PomaLoader`                            | `Doc2PomaReader`                        |
| **Sentence splitter** | `PomaSentenceSplitter`                    | `PomaSentenceNodeParser`                |
| **Chunkset splitter** | `PomaChunksetSplitter`                    | `PomaChunksetNodeParser`                |
| **Cheatsheet builder** | `PomaCheatsheetRetriever`                | `PomaCheatsheetPostProcessor`           |

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
| **poma-chunker** | Build depth-aware chunks & chunksets | [PyPI](https://pypi.org/project/poma-chunker/) | (private) |
| **poma-integrations** | Drop-in classes for LangChain / LlamaIndex | [PyPI](https://pypi.org/project/poma-integrations/) | [GitHub](https://github.com/poma-science/poma-integrations) |

---

## LangChain Integration – Minimal Example & Workflow

```python
from poma_integrations.langchain_poma import Doc2PomaLoader, PomaChunksetSplitter, PomaCheatsheetRetriever

# 1. Convert and chunk your document
poma_doc = Doc2PomaLoader(POMA_CONFIG).load(INPUT_PATH)
poma_doc_id, docs, chunks, chunksets = PomaChunksetSplitter(POMA_CONFIG).split_documents(poma_doc)
# save chunks and chunksets for later retrieval
# embed docs in a vector store

# 2. Build a retriever and chain (see example.py for full implementation)
retriever = PomaCheatsheetRetriever(my_vector_store, my_chunks_fetcher, my_chunkset_fetcher)
```

**Typical workflow:**
1. Load and **convert** documents with `Doc2PomaLoader`.
2. **Split** the document into chunks / chunksets with `PomaChunksetSplitter`.
3. **Store** chunks / chunksets using your own callbacks.
4. Create a **retriever** (e.g. `PomaCheatsheetRetriever`) with your own callbacks and plug into your favorite QA or RAG pipeline.

➡️ _See [`src/poma_integrations/langchain_poma.py`](./src/poma_integrations/langchain_poma.py) for the full LangChain integration code_

---

## LlamaIndex Integration – Minimal Example & Workflow

```python
from poma_integrations.llamaindex_poma import Doc2PomaReader, PomaChunksetNodeParser, PomaCheatsheetPostProcessor

# 1. Convert and chunk your document
poma_doc = Doc2PomaReader(POMA_CONFIG).load_data(INPUT_PATH)
poma_doc_id, nodes, raw_chunks, chunksets = PomaChunksetNodeParser(POMA_CONFIG).get_nodes_from_documents(poma_doc)
# save chunks and chunksets for later retrieval
# embed nodes in a vector store

# 2. Build a post-processor and query engine (see example_llamaindex.py for full implementation)
post_processor = PomaCheatsheetPostProcessor(
    chunk_fetcher=my_chunks_fetcher,
    chunkset_fetcher=my_chunkset_fetcher,
)
query_engine = CheatsheetQueryEngine(
    retriever=index.as_retriever(),
    cheatsheet_processor=post_processor,
    llm=OpenAI(model=LLM_MODEL),
    prompt_template=PromptTemplate("Use the following context to answer the question.\n\n{context}\n\nQuestion: {question}"),
)
```

**Typical workflow:**
1. Load and **convert** documents with `Doc2PomaReader`.
2. **Split** the document into chunks / chunksets with `PomaChunksetNodeParser`.
3. **Store** chunks / chunksets using your own callbacks.
4. **Retrieval**: Use `PomaCheatsheetPostProcessor` in your LlamaIndex query engine.

➡️ _See [`src/poma_integrations/llamaindex_poma.py`](./src/poma_integrations/llamaindex_poma.py) for the full LlamaIndex integration code_

---

## Main Classes

| Class                        | Description                                 |
|------------------------------|---------------------------------------------|
| `Doc2PomaLoader`             | Loads and converts documents (LangChain)    |
| `PomaSentenceSplitter`       | Splits documents into sentences             |
| `PomaChunksetSplitter`       | Splits a document into chunksets            |
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

---

Enjoy hierarchy-aware RAG with **one import** instead of hundreds of lines of
glue code.
