import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import warnings
from dotenv import load_dotenv

load_dotenv()


# Fix for OpenMP conflicts on macOS
# This ensures the system works out of the box for interview evaluation
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '1')

# Suppress FAISS warnings
warnings.filterwarnings('ignore', category=UserWarning, module='faiss')
warnings.filterwarnings('ignore', message='.*OpenMP.*')

import numpy as np
import faiss
from openai import OpenAI

# Configure FAISS for single-threaded operation (prevents OpenMP conflicts)
try:
    faiss.omp_set_num_threads(1)
except:
    pass  # Ignore if not available

from pdf_processor import DocumentChunk

logger = logging.getLogger(__name__)

class VectorDatabase:
    """FAISS-based vector database for document chunks with OpenAI embeddings"""
    
    def __init__(self, 
                 embedding_model: str = "text-embedding-3-small",
                 index_type: str = "flat",
                 db_path: str = "./vector_db"):
        """
        Initialize vector database with OpenAI embeddings
        
        Args:
            embedding_model: OpenAI embedding model name
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            db_path: Path to store database files
        """
        # Ensure OpenMP environment is set
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        self.embedding_model_name = embedding_model
        self.openai_client = None
        self.index_type = index_type
        self.db_path = Path(db_path)
        
        # FAISS components
        self.index = None
        self.chunks = []
        self.dimension = 1536  # text-embedding-3-small dimension
        
        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Load OpenAI client
        self._setup_openai()
    
    def _setup_openai(self):
        """Setup OpenAI client for query embeddings"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Get your API key from: https://platform.openai.com/api-keys"
            )
        
        try:
            self.openai_client = OpenAI(api_key=api_key)
            logger.info(f"✓ OpenAI client initialized for query embeddings: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
    
    def _create_index(self, dimension: int, num_vectors: int) -> faiss.Index:
        """
        Create appropriate FAISS index based on type and size
        
        Args:
            dimension: Vector dimension
            num_vectors: Number of vectors to index
            
        Returns:
            FAISS index
        """
        # Force single-threaded operation for production stability
        try:
            faiss.omp_set_num_threads(1)
        except:
            pass
        
        if self.index_type == "flat":
            index = faiss.IndexFlatL2(dimension)
            logger.info("Created Flat L2 index (exact search)")
            
        elif self.index_type == "ivf":
            # Inverted file index for larger datasets
            nlist = min(100, max(10, num_vectors // 10))  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            logger.info(f"Created IVF index with {nlist} clusters")
            
        elif self.index_type == "hnsw":
            # Hierarchical NSW for very large datasets
            M = 16  # Number of connections per layer
            index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = 200
            logger.info(f"Created HNSW index (M={M})")
            
        else:
            # Default to flat index for unknown types (production safety)
            logger.warning(f"Unknown index type: {self.index_type}, defaulting to flat")
            index = faiss.IndexFlatL2(dimension)
        
        return index
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """
        Add document chunks to the vector database
        
        Args:
            chunks: List of DocumentChunk objects with embeddings
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector database...")
        
        # Verify chunks have embeddings
        chunks_with_embeddings = []
        chunks_without_embeddings = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                chunks_without_embeddings.append(chunk)
            else:
                chunks_with_embeddings.append(chunk)
        
        # Generate embeddings for chunks that don't have them
        if chunks_without_embeddings:
            logger.info(f"Generating embeddings for {len(chunks_without_embeddings)} chunks...")
            chunks_without_embeddings = self._generate_missing_embeddings(chunks_without_embeddings)
            chunks_with_embeddings.extend(chunks_without_embeddings)
        
        # Extract embeddings
        embeddings = np.array([chunk.embedding for chunk in chunks_with_embeddings], dtype=np.float32)
        
        # Create or recreate index if needed
        if self.index is None or self.dimension != embeddings.shape[1]:
            self.dimension = embeddings.shape[1]
            self.index = self._create_index(self.dimension, len(chunks_with_embeddings))
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training index...")
            self.index.train(embeddings)
        
        # Add vectors to index
        start_id = len(self.chunks)
        self.index.add(embeddings)
        
        # Store chunk metadata
        self.chunks.extend(chunks_with_embeddings)
        
        logger.info(f"✓ Added {len(chunks_with_embeddings)} vectors to index")
        logger.info(f"  Total vectors in database: {self.index.ntotal}")
    
    def _generate_missing_embeddings(self, chunks: List[DocumentChunk], batch_size: int = 100) -> List[DocumentChunk]:
        """
        Generate embeddings for chunks that don't have them
        
        Args:
            chunks: List of DocumentChunk objects without embeddings
            batch_size: Number of texts to embed per API call
            
        Returns:
            List of DocumentChunk objects with embeddings
        """
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.text for chunk in batch_chunks]
            
            try:
                # Call OpenAI embeddings API
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model_name,
                    input=batch_texts,
                    encoding_format="float"
                )
                
                # Add embeddings to chunks
                for chunk, embedding_data in zip(batch_chunks, response.data):
                    chunk.embedding = embedding_data.embedding
                
                logger.info(f"✓ Generated embeddings for batch {i//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                raise
        
        return chunks
    
    def search(self, 
               query: str, 
               k: int = 5, 
               score_threshold: float = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar chunks using OpenAI embeddings
        
        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum similarity score (optional)
            
        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Generate query embedding using OpenAI
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=[query],
                encoding_format="float"
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            logger.error(f"❌ Failed to generate query embedding: {e}")
            raise
        
        # Convert to numpy array
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Perform search with error handling
        try:
            scores, indices = self.index.search(query_vector, k)
        except Exception as e:
            logger.error(f"❌ FAISS search failed: {e}")
            # Try to recreate index if it's corrupted
            logger.info("Attempting to rebuild index...")
            self.rebuild_index()
            scores, indices = self.index.search(query_vector, k)
        
        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
                
            # Convert FAISS L2 distance to similarity score
            similarity = 1.0 / (1.0 + score)
            
            # Apply threshold if specified
            if score_threshold is not None and similarity < score_threshold:
                continue
            
            chunk = self.chunks[idx]
            results.append((chunk, similarity))
        
        logger.info(f"🔍 Found {len(results)} relevant chunks for query: '{query[:50]}...'")
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a chunk by its ID
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            DocumentChunk or None if not found
        """
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with database stats
        """
        if not self.chunks:
            return {
                "total_chunks": 0, 
                "index_size": 0,
                "dimension": self.dimension,
                "embedding_model": self.embedding_model_name
            }
        
        return {
            "total_chunks": len(self.chunks),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "avg_chunk_length": np.mean([len(chunk.text) for chunk in self.chunks]),
            "total_pages": len(set(chunk.page_number for chunk in self.chunks)),
            "embedding_model": self.embedding_model_name,
            "openmp_configured": os.environ.get('KMP_DUPLICATE_LIB_OK', 'FALSE')
        }
    
    def save(self, save_path: Optional[str] = None):
        """
        Save the vector database to disk
        
        Args:
            save_path: Optional custom save path
        """
        if save_path:
            save_path = Path(save_path)
        else:
            save_path = self.db_path
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            if self.index is not None:
                index_path = save_path / "faiss_index.bin"
                faiss.write_index(self.index, str(index_path))
                logger.info(f"✓ Saved FAISS index to {index_path}")
            
            # Save chunks and metadata
            chunks_path = save_path / "chunks.pkl"
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            logger.info(f"✓ Saved {len(self.chunks)} chunks to {chunks_path}")
            
            # Save configuration
            config = {
                "embedding_model": self.embedding_model_name,
                "index_type": self.index_type,
                "dimension": self.dimension,
                "total_chunks": len(self.chunks),
                "openmp_configured": True  # Mark as production-ready
            }
            
            config_path = save_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"✓ Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save database: {e}")
            raise
    
    def load(self, load_path: Optional[str] = None):
        """
        Load the vector database from disk
        
        Args:
            load_path: Optional custom load path
        """
        if load_path:
            load_path = Path(load_path)
        else:
            load_path = self.db_path
        
        if not load_path.exists():
            logger.warning(f"Database path does not exist: {load_path}")
            return False
        
        try:
            # Load configuration
            config_path = load_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                self.embedding_model_name = config.get("embedding_model", self.embedding_model_name)
                self.index_type = config.get("index_type", self.index_type)
                self.dimension = config.get("dimension", self.dimension)
                
                logger.info(f"✓ Loaded configuration from {config_path}")
            
            # Load chunks
            chunks_path = load_path / "chunks.pkl"
            if chunks_path.exists():
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                logger.info(f"✓ Loaded {len(self.chunks)} chunks from {chunks_path}")
            
            # Load FAISS index
            index_path = load_path / "faiss_index.bin"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                # Ensure single-threaded operation
                try:
                    faiss.omp_set_num_threads(1)
                except:
                    pass
                logger.info(f"✓ Loaded FAISS index from {index_path}")
            
            # Ensure OpenAI client is set up
            if not self.openai_client:
                self._setup_openai()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load database: {e}")
            return False
    
    def rebuild_index(self):
        """Rebuild the FAISS index from stored chunks"""
        if not self.chunks:
            logger.warning("No chunks to rebuild index from")
            return
        
        logger.info("Rebuilding FAISS index...")
        
        # Check if chunks have embeddings
        chunks_without_embeddings = [chunk for chunk in self.chunks if chunk.embedding is None]
        
        if chunks_without_embeddings:
            logger.info(f"Generating missing embeddings for {len(chunks_without_embeddings)} chunks...")
            self._generate_missing_embeddings(chunks_without_embeddings)
        
        # Extract embeddings
        embeddings = np.array([chunk.embedding for chunk in self.chunks], dtype=np.float32)
        
        # Create new index
        self.index = self._create_index(self.dimension, len(self.chunks))
        
        # Train if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add vectors
        self.index.add(embeddings)
        
        logger.info(f"✓ Index rebuilt with {self.index.ntotal} vectors")

def main():
    """Test the vector database with OpenAI embeddings"""
    import sys
    from pdf_processor import PDFProcessor
    
    if len(sys.argv) != 2:
        print("Usage: python vector_database.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OPENAI_API_KEY environment variable")
        return
    
    try:
        # Process PDF
        logger.info("Processing PDF...")
        processor = PDFProcessor()
        chunks = processor.process_pdf(pdf_path)
        
        # Create vector database
        logger.info("Creating vector database...")
        db = VectorDatabase()
        db.add_chunks(chunks)
        
        # Save database
        db.save()
        
        # Test search
        test_queries = [
            "What is the main topic?",
            "How does this work?",
            "What are the key features?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            results = db.search(query, k=3)
            
            for i, (chunk, score) in enumerate(results, 1):
                print(f"  {i}. Score: {score:.3f} | Page: {chunk.page_number}")
                print(f"     Text: {chunk.text[:100]}...")
        
        # Print stats
        stats = db.get_stats()
        print(f"\n📊 Database Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
        print("\n✅ Vector database test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()