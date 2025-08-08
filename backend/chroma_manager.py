# ChromaDB Manager - Production Version

# Fix SQLite issues FIRST - before any other imports
import sys
try:
    __import__('pysqlite3')  # type: ignore[reportMissingImports]
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✅ Successfully replaced sqlite3 with pysqlite3-binary")
except ImportError:
    print("❌ pysqlite3-binary not available, using system sqlite3")

# Additional SQLite check to ensure proper version
try:
    import sqlite3
    sqlite_version = sqlite3.sqlite_version
    print(f"🔍 SQLite version being used: {sqlite_version}")
    if sqlite_version < "3.35.0":
        print(f"❌ SQLite version {sqlite_version} is too old for ChromaDB (needs >= 3.35.0)")
        # Try to import pysqlite3 directly as sqlite3
        try:
            import pysqlite3.dbapi2 as sqlite3  # type: ignore[reportMissingImports]
            sys.modules['sqlite3'] = sqlite3
            print(f"✅ Forced pysqlite3 as sqlite3, new version: {sqlite3.sqlite_version}")
        except ImportError:
            raise Exception(f"Cannot use SQLite version {sqlite_version} with ChromaDB. Please install pysqlite3-binary.")
    else:
        print(f"✅ SQLite version {sqlite_version} is compatible with ChromaDB")
except Exception as e:
    print(f"❌ SQLite setup error: {e}")
    raise

import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global cache to prevent re-initialization
_CACHED_MANAGER = None

class ChromaDBManager:
    """ChromaDB with proper semantic embeddings and caching"""
    
    def __init__(self):
        global _CACHED_MANAGER
        if _CACHED_MANAGER is not None:
            # Return cached instance
            self.collection = _CACHED_MANAGER.collection
            logger.info("✅ Using cached ChromaDB instance")
            return
            
        self.collection = None
        self._setup()
        _CACHED_MANAGER = self  # Cache this instance
    
    def _setup(self):
        """ChromaDB setup with embeddings - only runs once"""
        # Ensure SQLite replacement is active before importing chromadb
        try:
            import sqlite3
            if sqlite3.sqlite_version < "3.35.0":
                print(f"❌ System SQLite {sqlite3.sqlite_version} too old, forcing pysqlite3...")
                import pysqlite3.dbapi2 as sqlite3  # type: ignore[reportMissingImports]
                import sys
                sys.modules['sqlite3'] = sqlite3
                print(f"✅ Using pysqlite3 version {sqlite3.sqlite_version}")
        except Exception as e:
            print(f"SQLite setup warning: {e}")
        
        import chromadb
        from chromadb.utils import embedding_functions
        import os
        
        logger.info("🔄 Initializing ChromaDB...")
        
        # Use persistent storage instead of in-memory
        db_path = "backend/chroma_db"
        os.makedirs(db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Use sentence transformers for better embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # Fast, good quality
        )
        
        try:
            # Try to get existing collection first
            self.collection = self.client.get_collection(
                name="cuny_programs",
                embedding_function=self.embedding_function
            )
            logger.info("✅ Found existing ChromaDB collection")
            # Check if collection has data
            count = self.collection.count()
            if count == 0:
                logger.info("📚 Collection is empty, reindexing...")
                self._load_and_index()
            else:
                logger.info(f"📊 Collection has {count} programs indexed")
        except:
            # Collection doesn't exist, create it
            try:
                self.collection = self.client.create_collection(
                    name="cuny_programs",
                    embedding_function=self.embedding_function
                )
                logger.info("✅ Created new ChromaDB collection")
                self._load_and_index()
            except Exception as e:
                logger.error(f"❌ Failed to create ChromaDB collection: {e}")
                raise
        
        logger.info("✅ ChromaDB with embeddings ready (cached for future use)")
    
    def get_or_create_collection(self, name: str):
        """Return a Chroma collection by name, creating it if missing, using the same embedding function."""
        try:
            return self.client.get_collection(name=name, embedding_function=self.embedding_function)
        except Exception:
            return self.client.create_collection(name=name, embedding_function=self.embedding_function)
    
    def _load_and_index(self):
        """Load and create semantic embeddings - only runs once"""
        with open('data/cleaned_programs.json', 'r') as f:
            programs = json.load(f)
        
        # Index all programs with rich text for embeddings
        docs = []
        metas = []
        ids = []
        
        logger.info(f"📚 Creating embeddings for {len(programs)} programs...")
        
        for i, prog in enumerate(programs):
            # Rich text for better semantic matching
            text = f"""
            Program: {prog.get('program_name', '')}
            College: {prog.get('college', '')}
            Degree: {prog.get('degree_type', '')}
            Field: {prog.get('cip_title', '')}
            Description: {prog.get('cip_title', '')}
            """.strip()
            
            docs.append(text)
            metas.append(prog)
            ids.append(str(i))
        
        # Add in larger batches for speed
        batch_size = 500  # Increased batch size
        for i in range(0, len(docs), batch_size):
            end_idx = min(i + batch_size, len(docs))
            
            self.collection.add(
                documents=docs[i:end_idx],
                metadatas=metas[i:end_idx],
                ids=ids[i:end_idx]
            )
            
            if i % 2000 == 0:  # Less frequent logging
                logger.info(f"📚 Indexed {end_idx} programs...")
        
        logger.info(f"✅ Indexed {len(docs)} programs with semantic embeddings")

    def reindex(self, force: bool = False) -> int:
        """Ensure the collection is indexed. If force=True, rebuild from scratch.

        Returns the number of items after (re)indexing.
        """
        try:
            count = self.collection.count()
        except Exception:
            count = 0

        if force:
            logger.info("🔄 Force reindex requested: dropping and recreating collection…")
            name = "cuny_programs"
            # Delete and recreate to avoid duplicate IDs
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name=name,
                embedding_function=self.embedding_function,
            )
            self._load_and_index()
            new_count = self.collection.count()
            logger.info(f"✅ Force reindex complete. {new_count} programs indexed.")
            return new_count

        # Non-force: only index if empty
        if count == 0:
            logger.info("📚 Collection empty; indexing now…")
            self._load_and_index()
            new_count = self.collection.count()
            logger.info(f"✅ Indexing complete. {new_count} programs indexed.")
            return new_count
        else:
            logger.info(f"ℹ️ Collection already indexed with {count} items; skipping reindex.")
            return count
    
    def is_available(self):
        """Check if ChromaDB is available for use"""
        return self.collection is not None
    
    def search_programs(self, query: str, n_results: int = 10):
        """Semantic search using embeddings"""
        if not self.collection:
            logger.error("ChromaDB collection not available")
            return {'metadatas': [[]], 'ids': [[]]}
        
        try:
            logger.info(f"🔍 Searching for: '{query}' (n_results={n_results})")
            results = self.collection.query(
                query_texts=[query], 
                n_results=n_results
            )
            logger.info(f"✅ Found {len(results['metadatas'][0]) if results['metadatas'] else 0} results")
            return {
                'metadatas': results['metadatas'],
                'ids': results['ids']
            }
        except Exception as e:
            logger.error(f"❌ ChromaDB search error: {type(e).__name__}: {str(e)}")
            return {'metadatas': [[]], 'ids': [[]]}
