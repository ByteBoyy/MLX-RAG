import utils.transport
import utils.session
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import PyPDF2
import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()



logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    text: str
    page_number: int
    chunk_id: str
    metadata: Dict
    embedding: Optional[List[float]] = None

class PDFProcessor:
    """Handles PDF loading, text extraction, and chunking with OpenAI embeddings"""

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize PDF processor with OpenAI embeddings

        Args:
            embedding_model: OpenAI embedding model name
        """
        self.embedding_model_name = embedding_model
        self.openai_client = None
        self.embedding_dimension = 1536  # text-embedding-3-small dimension
        self._setup_openai()

    def _setup_openai(self):
        """Setup OpenAI client for embeddings"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Get your API key from: https://platform.openai.com/api-keys"
            )
        
        try:
            self.openai_client = OpenAI(api_key=api_key)
            logger.info(f"✓ OpenAI client initialized for embeddings: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise

    def extract_text_pypdf2(self, pdf_path: str) -> List[Tuple[str, int]]:
        """
        Extract text using PyPDF2 (fallback method)

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of (text, page_number) tuples
        """
        pages = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append((text, page_num))

        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise

        return pages

    def extract_text_pymupdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        """
        Extract text using PyMuPDF (primary method)

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of (text, page_number) tuples
        """
        pages = []
        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                if text and text.strip():
                    pages.append((text, page_num + 1))

            doc.close()

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            raise

        return pages

    def load_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        """
        Load and extract text from PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of (text, page_number) tuples
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")

        logger.info(f"Loading PDF: {pdf_path}")

        # Try PyMuPDF first (better text extraction)
        try:
            pages = self.extract_text_pymupdf(str(pdf_path))
            logger.info(f"✓ Extracted {len(pages)} pages using PyMuPDF")
            return pages
        except Exception as e:
            logger.warning(f"PyMuPDF failed, trying PyPDF2: {e}")

        # Fallback to PyPDF2
        try:
            pages = self.extract_text_pypdf2(str(pdf_path))
            logger.info(f"✓ Extracted {len(pages)} pages using PyPDF2")
            return pages
        except Exception as e:
            logger.error(f"Both PDF extraction methods failed: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text

        Args:
            text: Raw text from PDF

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/]', '', text)

        # Fix common PDF extraction issues like ligatures
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')

        text = text.replace('’', "'")
        text = text.replace('‘', "'")
        text = text.replace('“', '"')
        text = text.replace('”', '"')

        return text.strip()

    def semantic_chunking(self, text: str, max_chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into chunks using semantic boundaries

        Args:
            text: Text to chunk
            max_chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping characters between chunks

        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If paragraph is too long, split by sentences
            if len(paragraph) > max_chunk_size:
                sentences = self._split_sentences(paragraph)

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            # Add overlap from the end of current chunk
                            if overlap > 0 and len(current_chunk) > overlap:
                                current_chunk = current_chunk[-overlap:] + " " + sentence
                            else:
                                current_chunk = sentence
                        else:
                            current_chunk = sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
            else:
                # Add entire paragraph if it fits
                if len(current_chunk) + len(paragraph) > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        # Add overlap
                        if overlap > 0 and len(current_chunk) > overlap:
                            current_chunk = current_chunk[-overlap:] + " " + paragraph
                        else:
                            current_chunk = paragraph
                    else:
                        current_chunk = paragraph
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be improved with NLTK
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, pages: List[Tuple[str, int]],
                     max_chunk_size: int = 512,
                     overlap: int = 50) -> List[DocumentChunk]:
        """
        Create document chunks from extracted pages

        Args:
            pages: List of (text, page_number) tuples
            max_chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks

        Returns:
            List of DocumentChunk objects
        """
        all_chunks = []
        chunk_counter = 0

        for page_text, page_num in pages:
            # Clean the text
            cleaned_text = self.clean_text(page_text)

            if not cleaned_text:
                continue

            # Create semantic chunks
            text_chunks = self.semantic_chunking(cleaned_text, max_chunk_size, overlap)

            for chunk_text in text_chunks:
                chunk_id = f"chunk_{chunk_counter:04d}"

                chunk = DocumentChunk(
                    text=chunk_text,
                    page_number=page_num,
                    chunk_id=chunk_id,
                    metadata={
                        'length': len(chunk_text),
                        'page': page_num,
                        'chunk_index': chunk_counter
                    }
                )

                all_chunks.append(chunk)
                chunk_counter += 1

        logger.info(f"✓ Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks

    def generate_embeddings(self, chunks: List[DocumentChunk], batch_size: int = 100) -> List[DocumentChunk]:
        """
        Generate embeddings for all chunks using OpenAI API

        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of texts to embed per API call

        Returns:
            List of DocumentChunk objects with embeddings
        """
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")

        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.embedding_model_name}...")

        # Process chunks in batches to avoid API limits
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.text for chunk in batch_chunks]
            
            try:
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
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

        logger.info("✓ All embeddings generated successfully")
        return chunks

    def generate_embeddings_with_progress(self, chunks: List[DocumentChunk], 
                                        batch_size: int = 100, 
                                        progress_callback=None) -> List[DocumentChunk]:
        """
        Generate embeddings with progress callback support
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of texts to embed per API call
            progress_callback: Function to call after each batch
            
        Returns:
            List of DocumentChunk objects with embeddings
        """
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")

        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.embedding_model_name}...")

        # Process chunks in batches to avoid API limits
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
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback()
                
                logger.debug(f"✓ Generated embeddings for batch {i//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                raise

        logger.info("✓ All embeddings generated successfully")
        return chunks

   
    def process_pdf(self, pdf_path: str,
                   max_chunk_size: int = 512,
                   overlap: int = 50,
                   batch_size: int = 100) -> List[DocumentChunk]:
        """
        Complete PDF processing pipeline with OpenAI embeddings

        Args:
            pdf_path: Path to PDF file
            max_chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks
            batch_size: Batch size for embedding generation

        Returns:
            List of DocumentChunk objects with embeddings
        """
        logger.info(f"🔄 Processing PDF: {pdf_path}")

        # Load PDF and extract text
        pages = self.load_pdf(pdf_path)

        # Create chunks
        chunks = self.create_chunks(pages, max_chunk_size, overlap)

        # Generate embeddings using OpenAI
        chunks_with_embeddings = self.generate_embeddings(chunks, batch_size)

        logger.info(f"✅ PDF processing complete: {len(chunks_with_embeddings)} chunks created")
        return chunks_with_embeddings

def main():
    """Test the PDF processor with OpenAI embeddings"""
    import sys

    # Create a dummy PDF for testing if no path is provided
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        dummy_pdf_path = "dummy_document.pdf"
        if not os.path.exists(dummy_pdf_path):
            c = canvas.Canvas(dummy_pdf_path, pagesize=letter)
            c.drawString(100, 750, "This is a test document with OpenAI embeddings.")
            c.drawString(100, 730, "It demonstrates the integration with 'smart quotes' and “double quotes”.")
            c.showPage()
            c.save()
            print(f"Created a dummy PDF for testing: {dummy_pdf_path}")
        pdf_path = dummy_pdf_path
    except ImportError:
        print("Please provide a PDF path or run 'pip install reportlab' to create a dummy file.")
        return

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY=your_api_key_here")
        return

    # Process PDF
    processor = PDFProcessor()
    chunks = processor.process_pdf(pdf_path)

    # Print summary
    print(f"\n📊 Processing Summary:")
    if chunks:
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Average chunk length: {sum(len(c.text) for c in chunks) / len(chunks):.1f} chars")
        print(f"   Pages covered: {max(c.page_number for c in chunks)}")
        print(f"   Embedding dimension: {len(chunks[0].embedding) if chunks[0].embedding else 'None'}")

        # Show first chunk as example
        print(f"\n📝 First chunk example:")
        print(f"   ID: {chunks[0].chunk_id}")
        print(f"   Page: {chunks[0].page_number}")
        print(f"   Text: {chunks[0].text[:200]}...")
        print(f"   Embedding sample: {chunks[0].embedding[:5] if chunks[0].embedding else 'None'}...")
    else:
        print("   No chunks were generated.")

if __name__ == "__main__":
    main()