"""
PDF Client for downloading, parsing, chunking, embedding, and querying PDF documents
via vector similarity using Qdrant.

Pipeline:
1. Download PDFs from URLs
2. Parse pages: text + (filename, page_number, section_header)
3. Chunk pages into retrievable units
4. Embed chunks
5. Store vectors + metadata in Qdrant
6. Query by semantic similarity, optionally filtered by URL
"""

import os
import uuid
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import logging

from veris_chat.ingestion.embed_core import Embedder
from veris_chat.ingestion.download_core import process_url as download_pdf_url
from veris_chat.ingestion.parse_core import process_pdf
from veris_chat.ingestion.chunk_core import chunk_pages

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        PayloadSchemaType,
    )
except ImportError as e:
    raise ImportError(
        f"Missing required dependencies for Client: {e}. "
        "Install with: pip install qdrant-client sentence-transformers pymupdf requests"
    )

logger = logging.getLogger(__name__)


class IngestionClient:
    """Client for PDF document ingestion (to Qdrant) and vector-based querying."""

    def __init__(
        self,
        storage_path: str = "./qdrant_local",
        collection_name: str = "veris_pdfs",
        embedding_model: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        embedding_dim: int =None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize PDF client with vector storage.

        Args:
            storage_path: Path to store local Qdrant database (used if QDRANT_URL not set).
            collection_name: Name of the Qdrant collection.
            embedding_model: Embedding model ID for Embedder (Bedrock or HF).
            chunk_size: Maximum characters per chunk (used in fixed-size chunking).
            chunk_overlap: Overlap between consecutive chunks (fixed-size chunking).
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

        self.encoder: Optional[Embedder] = None
        self.embedding_dim: Optional[int] = embedding_dim

        # PDF storage directory for downloaded files
        self.pdf_dir = Path.cwd() / "data" / self.collection_name
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Qdrant client (cloud or local)
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if qdrant_url:
            logger.info(f"[QDRANT] Using remote Qdrant instance at {qdrant_url}")
            self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            logger.info("[QDRANT] Using local Qdrant (embedded) storage.")
            self.qdrant = QdrantClient(path=str(self.storage_path))

        # Track processed URLs and session mappings
        self.cache_file = self.pdf_dir / "url_cache.json"
        self.session_index_file = self.pdf_dir / "session_index.json"
        self.url_cache: Dict[str, Any] = {}  # url → {local_path, ingestion_time, ...}
        self.session_index: Dict[str, Set[str]] = {}  # session_id → Set[url]
        self._load_url_cache()
        self._load_session_index()
        
        if self.embedding_dim:
            self._ensure_collection()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _initialize_embedder(self):
        """Lazily initialize the Embedder and embedding dimension."""
        if self.encoder:
            return

        print("\n" + "=" * 70)
        print(f"Initializing embedding model: {self.embedding_model}")
        print("=" * 70)

        self.encoder = Embedder(
            model=self.embedding_model,
            region=os.getenv("AWS_REGION", "us-east-1"),
        )

        print("=" * 70)
        print("Embedder initialization complete!")
        print("=" * 70 + "\n")
        
        if self.embedding_dim is None:
            self.embedding_dim = self.encoder.embedding_dim
            # Ensure collection now that we know embedding_dim
            self._ensure_collection()
        else:
            assert self.embedding_dim == self.encoder.embedding_dim, f"Mismatch embedding_dim."

    def reset_collection(self, delete_pdfs: bool = False) -> Dict[str, Any]:
        """
        Delete the Qdrant collection and clear all related cache files.
        
        This is useful for testing or when you need a completely fresh start.
        
        Args:
            delete_pdfs: If True, also delete downloaded PDF files in pdf_dir.
                        Default False to preserve downloaded files.
        
        Clears:
            - Qdrant collection (deleted and recreated empty)
            - url_cache (in-memory + url_cache.json)
            - session_index (in-memory + session_index.json)
            - Optionally: PDF files in pdf_dir
            
        Returns:
            Dict with reset metadata: {elapsed_time, deleted_pdfs_count}
        """
        t_start = time.perf_counter()
        deleted_pdfs_count = 0
        
        logger.info(f"[RESET] Resetting collection '{self.collection_name}'...")
        
        # 1. Delete Qdrant collection
        try:
            self.qdrant.delete_collection(collection_name=self.collection_name)
            logger.info(f"[RESET] Deleted Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"[RESET] Collection may not exist: {e}")
        
        # 2. Clear url_cache (in-memory + file)
        self.url_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
            logger.info(f"[RESET] Deleted url_cache file: {self.cache_file}")
        
        # 3. Clear session_index (in-memory + file)
        self.session_index.clear()
        if self.session_index_file.exists():
            self.session_index_file.unlink()
            logger.info(f"[RESET] Deleted session_index file: {self.session_index_file}")
        
        # 4. Optionally delete PDF files
        if delete_pdfs and self.pdf_dir.exists():
            # Delete all PDF files but keep the directory
            for pdf_file in self.pdf_dir.glob("*.pdf"):
                pdf_file.unlink()
                deleted_pdfs_count += 1
                logger.debug(f"[RESET] Deleted PDF: {pdf_file.name}")
            logger.info(f"[RESET] Deleted {deleted_pdfs_count} PDFs in {self.pdf_dir}")
        
        # 5. Recreate empty collection with proper config
        if self.embedding_dim is None:
            self._initialize_embedder() # This will also call _ensure_collection()
        else:
            self._ensure_collection()
        
        elapsed_time = time.perf_counter() - t_start
        logger.info(f"[RESET] ⏱ ({elapsed_time:.3f}s) Collection reset complete")
        
        return {
            "elapsed_time": round(elapsed_time, 3),
            "deleted_pdfs_count": deleted_pdfs_count,
        }
        
    def _ensure_collection(self):
        """Create collection if it doesn't exist and set up indexes."""
        logger.info("[QDRANT] Ensure collection")
        if self.embedding_dim is None:
            raise ValueError(
                "Embedding dimension is not set. Call _initialize_embedder() first."
            )

        collections = self.qdrant.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if collection_exists:
            logger.info(f"[QDRANT] Collection '{self.collection_name}' already exists.")
        else:
            logger.info(
                f"[QDRANT] Creating collection '{self.collection_name}' "
                f"with vector size={self.embedding_dim}"
            )
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
        
        # Create keyword index on 'url' for efficient filtered queries.
        # Note: session_id is now tracked in session_index, not in Qdrant payload
        try:
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="url",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info(f"[QDRANT] Created keyword index on 'url' field")
        except Exception as e:
            # Index might already exist, which is fine
            logger.debug(f"[QDRANT] Index creation for 'url' skipped or failed: {e}")
    
    # ------------------------------------------------------------------
    # Session Index: session_id → Set[url]
    # ------------------------------------------------------------------
    def _load_session_index(self):
        """Load session_id → Set[url] mapping from JSON."""
        if self.session_index_file.exists():
            try:
                with open(self.session_index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Convert lists back to sets
                self.session_index = {k: set(v) for k, v in data.items()}
                logger.info(f"[SESSION] Loaded {len(self.session_index)} sessions from {self.session_index_file}")
            except Exception as e:
                logger.error(f"[SESSION] Failed reading {self.session_index_file}: {e}")
                self.session_index = {}
        else:
            logger.info("[SESSION] No session_index.json found. Starting empty.")
            self.session_index = {}
    
    def _save_session_index(self):
        """Persist session_id → Set[url] mapping to JSON."""
        try:
            # Convert sets to lists for JSON serialization
            data = {k: list(v) for k, v in self.session_index.items()}
            with open(self.session_index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"[SESSION] Saved session index to {self.session_index_file}")
        except Exception as e:
            logger.error(f"[SESSION] Failed to save session index: {e}")
    
    def get_session_urls(self, session_id: str) -> Set[str]:
        """
        Get all URLs associated with a session.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Set of URLs for this session (empty set if session not found).
        """
        return self.session_index.get(session_id, set())
    
    def add_url_to_session(self, session_id: str, url: str) -> None:
        """
        Add a URL to a session's URL set.
        
        Args:
            session_id: Session identifier.
            url: URL to add to the session.
        """
        if session_id not in self.session_index:
            self.session_index[session_id] = set()
        self.session_index[session_id].add(url)
        
    def _save_url_cache(self):
        """Persist URL → local_path mapping to data/url_cache.json."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.url_cache, f, indent=2)
            logger.info(f"[CACHE] Saved URL cache to {self.cache_file}")
        except Exception as e:
            logger.error(f"[CACHE] Failed to save URL cache: {e}")


    def _load_url_cache(self):
        """Load URL→{local_path, ingestion_time} cache from JSON first, then merge Qdrant contents."""
        # 1) Load from local JSON
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.url_cache = json.load(f)
                logger.info(f"[CACHE] Loaded {len(self.url_cache)} entries from {self.cache_file}")
            except Exception as e:
                logger.error(f"[CACHE] Failed reading {self.cache_file}: {e}")
                self.url_cache = {}
        else:
            logger.info("[CACHE] No url_cache.json found. Starting empty cache.")
            self.url_cache = {}

        # 2) Merge Qdrant URLs if any (for URLs not in local cache)
        try:
            offset = None
            while True:
                records, offset = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for r in records:
                    payload = r.payload or {}
                    url = payload.get("url")
                    if url and url not in self.url_cache:
                        # Add minimal entry for URLs found in Qdrant but not in cache
                        self.url_cache[url] = {
                            "local_path": payload.get("local_path"),
                            "ingestion_time": None,
                            "ingested_at": None,
                        }
                if offset is None:
                    break
        except Exception as e:
            logger.warning(f"[CACHE] Failed to load URL metadata from Qdrant: {e}")

        logger.info(f"[CACHE] Total cached URLs after merge: {len(self.url_cache)}")

    # ------------------------------------------------------------------
    # Ingestion helpers
    # ------------------------------------------------------------------
    def _download_parse_pdf(self, url: str) -> List[dict]:
        """
        Download PDF from URL and parse into page-level records.

        Returns:
            List[dict]: each dict has keys:
                - filename
                - page_number
                - text
                - section_header
                - url
        """
        logger.info(f"[INGEST] Downloading and parsing PDF from URL: {url}")

        info = download_pdf_url(url, output_dir=str(self.pdf_dir))
        if info is None:
            raise RuntimeError(f"Failed to download PDF from {url}")

        local_path = info["local_path"]
        filename = info["filename"]

        parsed_pages = process_pdf(local_path) or []
        for p in parsed_pages:
            # Attach URL as metadata for downstream payload
            p["url"] = url
            # Ensure filename is consistent with downloaded file
            p["filename"] = filename

        logger.info(
            f"[INGEST] Parsed {len(parsed_pages)} pages from {filename} ({url})"
        )
        return parsed_pages

    def _index_document(self, url: str, parsed_pages: List[dict]):
        """
        Chunk parsed pages, embed chunks, and index into Qdrant.

        Args:
            url: Original document URL
            parsed_pages: List of parsed page dicts from _download_parse_pdf
        
        Note: session_id is NOT stored in Qdrant payload. Session-URL mapping
        is tracked separately in session_index for efficient multi-session support.
        """
        if not parsed_pages:
            logger.warning(f"[INGEST] No pages parsed for URL: {url}")
            return

        # Chunk pages (fixed-size or paragraph-based as configured)
        chunks = chunk_pages(
            parsed_pages,
            strategy="fixed",
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

        if not chunks:
            logger.warning(f"[INGEST] No chunks produced for URL: {url}")
            return

        texts = [c["text"] for c in chunks]
        logger.info(f"[EMBED] Embedding {len(texts)} chunks for URL: {url}")

        embeddings = self.encoder.embed(texts)
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding count {len(embeddings)} does not match chunk count {len(chunks)}"
            )

        # Prepare Qdrant points (no session_id in payload)
        points: List[PointStruct] = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}#{idx}"))
            payload = {
                "url": url,
                "filename": chunk.get("filename"),
                "page_number": chunk.get("page_number"),
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": idx,
                "section_header": chunk.get("section_header"),
                "text": chunk.get("text"),
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        logger.info(
            f"[QDRANT] Upserting {len(points)} points into collection '{self.collection_name}'"
        )
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        logger.info(f"[INGEST] Finished indexing document from URL: {url}")

    # ------------------------------------------------------------------
    # Public API: store & query
    # ------------------------------------------------------------------
    def store(self, url: str, session_id: Optional[str] = None):
        """
        Ingest a PDF document from URL into the vector store.

        Ingestion workflow:
        1. Always add URL to session_index for the given session_id
        2. Only run full ingestion (download, parse, embed, store) if URL not in url_cache
        
        This means: same URL in 2 sessions = 1x Qdrant storage (chunks are shared)

        Args:
            url: URL of the PDF document to ingest.
            session_id: Optional session identifier for session-scoped retrieval.
                        URL is tracked in session_index, NOT in Qdrant payload.
                        
        Returns:
            Dict with ingestion metadata: {local_path, ingestion_time, ingested_at, skipped}
        """
        self._initialize_embedder()

        # Step 1: Always add URL to session_index (even if already in url_cache)
        if session_id:
            self.add_url_to_session(session_id, url)
            self._save_session_index()
            logger.info(f"[SESSION] Added URL to session '{session_id}': {url}")
        
        # Step 2: Only ingest if URL not in url_cache
        if url in self.url_cache:
            logger.info(f"[INGEST] URL already ingested (chunks exist), skipping: {url}")
            return {**self.url_cache[url], "skipped": True}

        t_start = time.perf_counter()
        
        parsed_pages = self._download_parse_pdf(url)
        
        # Get local_path from the first parsed page (all pages share same file)
        local_path = None
        if parsed_pages:
            # Reconstruct local_path from pdf_dir and filename
            filename = parsed_pages[0].get("filename", "")
            local_path = str(self.pdf_dir / filename) if filename else None
        
        # Index document (no session_id in Qdrant payload)
        self._index_document(url, parsed_pages)
        
        ingestion_time = time.perf_counter() - t_start
        
        # Update url_cache with ingestion metadata
        self.url_cache[url] = {
            "local_path": local_path,
            "ingestion_time": round(ingestion_time, 3),
            "ingested_at": datetime.now().isoformat(),
        }
        self._save_url_cache()
        logger.info(f"[INGEST] ⏱ ({ingestion_time:.2f}s) URL ingested and cached: {url}")
        
        return {**self.url_cache[url], "skipped": False}

    def retrieve(self, query: str, url: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
        """
        Perform a similarity search over stored documents.

        Args:
            query: Search query string.
            url: If provided, restrict search to a single document URL.
            top_k: Number of top results to return.

        Returns:
            Dict containing query, url (if any), matched chunks with scores, and retrieval_time.
        """
        t_start = time.perf_counter()
        self._initialize_embedder()

        # Build Qdrant filter if URL is specified
        query_filter = None
        if url is not None:
            # without adding the keyword index added in `_ensure_collection`,
            # Qdrant Local: Scan all documents
            # Qdrant Cloud: Error
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="url",
                        match=MatchValue(value=url),
                    )
                ]
            )

        query_vec = self.encoder.embed([query])[0]

        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        chunks = []
        for result in response.points:
            payload = result.payload or {}
            chunks.append(
                {
                    "text": payload.get("text"),
                    "chunk_index": payload.get("chunk_index"),
                    "url": payload.get("url"),
                    "filename": payload.get("filename"),
                    "page_number": payload.get("page_number"),
                    "section_header": payload.get("section_header"),
                    "score": result.score,
                }
            )

        retrieval_time = time.perf_counter() - t_start
        logger.info(f"[RETRIEVE] ⏱ ({retrieval_time:.3f}s) Query completed: {len(chunks)} chunks")
        
        return {
            "query": query,
            "url": url,
            "chunks": chunks,
            "num_results": len(chunks),
            "retrieval_time": round(retrieval_time, 3),
        }

    def ping(self) -> bool:
        """Check if Qdrant client is accessible."""
        try:
            self.qdrant.get_collections()
            return True
        except Exception as e:
            logger.error(f"[QDRANT] Ping failed: {e}")
            return False
