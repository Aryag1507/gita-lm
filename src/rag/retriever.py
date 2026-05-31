"""
Retrieval-Augmented Generation layer.

At index time: embed all commentary chunks using a local sentence-transformer
model and store them in ChromaDB on disk.

At inference time: embed the query verse, retrieve the top-k most semantically
similar commentaries, and prepend them as context before generation.

This implements non-parametric retrieval on top of parametric fine-tuning.
The two approaches are complementary:
  - Fine-tuning bakes Prabhupada's style and vocabulary into the model weights
  - RAG supplies relevant factual content the model may not have memorized
"""

import sqlite3

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import DataConfig, RAGConfig
from src.data.dataset import CommentaryRecord


def _load_all_records(data_config: DataConfig) -> list[CommentaryRecord]:
    conn = sqlite3.connect(data_config.db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT v.chapter, v.verse_label, v.sanskrit, v.translation, c.text, c.acharya
        FROM commentaries c
        JOIN verses v ON c.verse_id = v.id
        ORDER BY v.chapter, v.verse
        """
    )
    rows = cur.fetchall()
    conn.close()

    records = []
    for chapter, verse_label, sanskrit, translation, commentary, acharya in rows:
        if commentary and commentary.strip():
            records.append(
                CommentaryRecord(
                    chapter=chapter,
                    verse_label=verse_label,
                    sanskrit=(sanskrit or "").strip(),
                    translation=(translation or "").strip(),
                    commentary=commentary.strip(),
                )
            )
    return records


class CommentaryRetriever:
    def __init__(self, rag_config: RAGConfig, data_config: DataConfig):
        self.config = rag_config
        self.data_config = data_config
        self.embedder = SentenceTransformer(rag_config.embedding_model)
        self.client = chromadb.PersistentClient(
            path=rag_config.chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=rag_config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, force: bool = False) -> None:
        """Embed all commentaries and store in ChromaDB."""
        if self.collection.count() > 0 and not force:
            print(f"Index already contains {self.collection.count()} documents. Use force=True to re-index.")
            return

        print("Loading records from database...")
        records = _load_all_records(self.data_config)

        texts, ids, metadatas = [], [], []
        for i, record in enumerate(records):
            # Index the verse context + commentary together so retrieval
            # finds commentaries that are semantically relevant to the query
            text = (
                f"BG {record.chapter}.{record.verse_label} "
                f"Translation: {record.translation} "
                f"Commentary: {record.commentary[:500]}"
            )
            texts.append(text)
            ids.append(f"verse_{record.chapter}_{record.verse_label}_{i}")
            metadatas.append({
                "chapter": record.chapter,
                "verse_label": record.verse_label,
                "translation": record.translation[:200],
                "commentary_preview": record.commentary[:300],
            })

        print(f"Embedding {len(texts)} documents...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()

        # Upsert in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        print(f"Indexed {self.collection.count()} documents into ChromaDB")

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Given a verse description or question, return the top-k most
        semantically similar commentaries as context strings.
        """
        k = top_k or self.config.top_k
        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "text": doc,
                "chapter": meta["chapter"],
                "verse_label": meta["verse_label"],
                "translation": meta["translation"],
                "commentary_preview": meta["commentary_preview"],
                "similarity": 1 - dist,  # cosine distance → similarity
            })

        return retrieved

    def build_rag_prompt(self, chapter: int, verse_label: str, translation: str, sanskrit: str) -> str:
        """
        Retrieve relevant context and build an augmented prompt.
        The retrieved commentaries give the model relevant factual
        content to draw on, while fine-tuning provides the style.
        """
        query = f"BG {chapter}.{verse_label} {translation}"
        retrieved = self.retrieve(query)

        context_block = "\n\n".join([
            f"[Context from BG {r['chapter']}.{r['verse_label']}]: {r['commentary_preview']}"
            for r in retrieved
        ])

        return (
            f"### Relevant Commentaries:\n{context_block}\n\n"
            f"### Verse BG {chapter}.{verse_label}\n"
            f"Sanskrit: {sanskrit}\n"
            f"Translation: {translation}\n"
            f"### Commentary:\n"
        )
