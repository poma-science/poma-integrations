# 🍍 POMA ↔ LangChain & Llama-Index mini-SDK

[![PyPI version](https://img.shields.io/pypi/v/poma-integrations.svg)](https://pypi.org/project/poma-integrations/)  
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](LICENSE)

> **Part of the POMA toolkit.** For the complete documentation, see the [organization README](https://github.com/poma-science/.github).

Turn **any** PDF / HTML / image into
*clean markdown*, **structure-preserving chunksets**, and
token-efficient **cheatsheets** – all with the native extension points
you already know from each framework.

| Layer               | LangChain class                             | Llama-Index class                       |
|---------------------|---------------------------------------------|-----------------------------------------|
| **Loader / Reader** | `Doc2PomaLoader`                            | `Doc2PomaReader`                        |
| **Sentence splitter** | `PomaSentenceSplitter`                    | `PomaSentenceNodeParser`                |
| **Chunkset splitter** | `PomaChunksetSplitter`                    | `PomaChunksetNodeParser`                |
| **Cheatsheet builder** | `PomaCheatsheetRetriever`                | `PomaCheatsheetPostProcessor`           |

The *only* thing **you** must supply is a **`chunk_store`** and a **`chunk_fetcher` callback**
that returns raw sentences given `(doc_id, chunk_ids)`.
Store them in SQLite, Postgres, Redis, S3 – whatever fits your stack.

---

## Quick-Start – LangChain version

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import sys
from poma_integrations import Doc2PomaLoader, PomaSentenceSplitter, PomaChunksetSplitter, PomaCheatsheetRetriever
import sqlite3

# Required packages for langchain POMA:
# `pip install poma-integrations`
# Required additional packages for this example:
# `pip install langchain-openai chromadb sentence-transformers`
# These provide the LLM interface, vector store, and embedding model.


# Configuration for document conversion and chunking using LiteLLM-compatible providers and models.
cfg = {
    "conversion_provider": "gemini",
    "conversion_model": "gemini-2.0-flash",
    "chunking_provider": "openai",
    "chunking_model": "gpt-4.1-mini",
}

# 1️⃣ ingest
# Convert the document to a .poma archive using the Doc2PomaLoader.
loader = Doc2PomaLoader(cfg)
docs_md = loader.load("bopa_Llei_example.pdf")

# Split the document into structured chunksets and raw text chunks using the configured chunking model.
chunksets, raw_chunks = PomaChunksetSplitter(cfg).split_documents(docs_md)


# Store the chunks in a SQLite database (or another key-value store / db) for later retrieval.
con = sqlite3.connect("chunks.db")

def chunk_store(con, chunksets, raw_chunks):
    """
    Store the chunksets in a SQLite database.
    """
    con.executescript(
        "CREATE TABLE IF NOT EXISTS chunks(doc_id TEXT, idx INT, depth INT, "
        "content TEXT, PRIMARY KEY(doc_id,idx));"
    )
    doc_id = chunksets[0].metadata["doc_id"]
    con.executemany(
        "INSERT OR IGNORE INTO chunks VALUES (?,?,?,?)",
        [(doc_id, c["chunk_index"], c["depth"], c["content"]) for c in raw_chunks],
    )
    con.commit()

# Store the chunksets and raw chunks in the SQLite database.
chunk_store(con, chunksets, raw_chunks)

# Fetch all chunks for a given document ID from the SQLite database.
def chunk_fetcher(doc_id, con=con):
    """
    Return *all* chunks belonging to the document.
    """
    rows = con.execute(
        "SELECT idx, depth, content FROM chunks WHERE doc_id=? ORDER BY idx",
        (doc_id,),
    ).fetchall()
    return [
        {"chunk_index": r[0], "depth": r[1], "content": r[2]}
        for r in rows
    ]

# Create a vector store from the Poma chunksets using HuggingFace embeddings.
# You can use any other vector DB and embedding provider.
vec = Chroma.from_documents(chunksets, HuggingFaceEmbeddings())


# 2️⃣ Retrieve: Initialize the retriever and QA chain
retriever = PomaCheatsheetRetriever(vec, chunk_fetcher)

# Create a RetrievalQA chain with the retriever and your preferred language model.
llm = ChatOpenAI(model="gpt-4o")
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)

# Query the chain to get an answer based on information retrieved from the cheatsheet.
print(chain("How much is a vanity plate including all probable fees?"))

```

## Quick-Start – LlamaIndex version

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.openai import OpenAI
import sys
from poma_integrations import Doc2PomaReader, PomaSentenceNodeParser, PomaChunksetNodeParser, PomaCheatsheetPostProcessor
import sqlite3

# Required packages for llamaindex POMA:
# `pip install poma-integrations`


# Use the following configuration to set up the reader and chunking, or any other providers and models from LiteLLM.
cfg = {
    "conversion_provider": "gemini",
    "conversion_model": "gemini-2.0-flash",
    "chunking_provider": "openai",
    "chunking_model": "gpt-4.1-mini",
}

reader = Doc2PomaReader(cfg)

# 1️⃣ ingest
# Chunk the document and convert it to Poma nodes.
nodes_md = reader.load_data("bopa_Llei_example.pdf")
nodes, raw_chunks = PomaChunksetNodeParser(cfg).get_nodes_from_documents(nodes_md)


# Store/Fetch the chunks in a SQLite database, or any prefered key-value DB for later retrieval.
con = sqlite3.connect("chunks.db")

def chunk_store(con, nodes, raw_chunks):
    """
    Store the chunksets in a SQLite database.
    """
    # Create a table for the chunks if it does not exist.
    con.executescript(
        "CREATE TABLE IF NOT EXISTS chunks(doc_id TEXT, idx INT, depth INT, "
        "content TEXT, PRIMARY KEY(doc_id,idx));"
    )
    doc_id = nodes[0].metadata["doc_id"]
    con.executemany(
        "INSERT OR IGNORE INTO chunks VALUES (?,?,?,?)",
        [(doc_id, c["chunk_index"], c["depth"], c["content"]) for c in raw_chunks],
    )
    con.commit()

# Store the chunks in the SQLite database.
chunk_store(con, nodes, raw_chunks)

# Define a function to fetch all chunks from a document from the SQLite database.
def chunk_fetcher(doc_id, con=con):
    """
    Return *all* chunks belonging to the document.
    """
    rows = con.execute(
        "SELECT idx, depth, content FROM chunks WHERE doc_id=? ORDER BY idx",
        (doc_id,),
    ).fetchall()
    return [
        {"chunk_index": r[0], "depth": r[1], "content": r[2]}
        for r in rows
    ]

# Create a VectorStoreIndex from the Poma nodes, or any other index type / vector database.
index = VectorStoreIndex(nodes)


# 2️⃣ retrieve 

# Create a post-processor that fetches the chunks from the key-value database, and returns a convinient cheatsheet.
post = PomaCheatsheetPostProcessor(chunk_fetcher = chunk_fetcher)

# Create a query engine with the post-processor.
qe = index.as_query_engine(node_postprocessors=[post], llm = OpenAI(model="gpt-4.1-nano"), chat_mode=ChatMode.REACT)

# Now you can query the index and get a cheatsheet with the results.
print(qe.query("How much is a vanity plate including all probable fees?"))
```




---

## 🚀 Installation

```bash
 `pip install .`
```

*Optionally add `langchain` or `llama-index-core` to the same command.*

---

Enjoy hierarchy-aware RAG with **one import** instead of hundreds of lines of
glue code.
