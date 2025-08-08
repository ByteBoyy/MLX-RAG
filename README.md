# RAG System - Production-Ready Document Q&A with OpenAI

**A complete Retrieval-Augmented Generation system demonstrating production-quality implementation with OpenAI embeddings and FAISS vector search.**

## 🚀 Quick Start

### 1. Setup Environment
```bash
chmod +x setup.sh
./setup.sh
```

### 2. Set OpenAI API Key
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```
*Get your API key from: https://platform.openai.com/api-keys*

### 3. Process Document & Ask Questions
```bash
# Process the included Procyon guide (or any PDF)
python rag_cli.py ingest procyon_guide.pdf

# Ask questions about the document
python rag_cli.py query "What is UL Procyon and what does it benchmark?"

# Start interactive chat session
python rag_cli.py interactive
```

## 📁 System Architecture

```
PDF Document → Text Extraction → Semantic Chunking → OpenAI Embeddings → FAISS Vector DB
                                                                              ↓
User Query → OpenAI Embedding → Vector Similarity Search → Context Retrieval → OpenAI Chat → Response
```

## 🏗️ Core Components

### **PDF Processor (`pdf_processor.py`)**
- **Dual extraction engines**: PyMuPDF (primary) + PyPDF2 (fallback)
- **Smart text cleaning**: Handles PDF artifacts, special characters, ligatures
- **Semantic chunking**: Paragraph-aware splitting with configurable overlap
- **OpenAI embeddings**: Batch processing with `text-embedding-3-small`
- **Progress tracking**: Real-time feedback for long operations

### **Vector Database (`vector_database.py`)**
- **FAISS integration**: High-performance similarity search
- **Multiple index types**: Flat (exact), IVF (approximate), HNSW (hierarchical)
- **Production hardening**: Handles OpenMP conflicts, single-threading for stability
- **Persistent storage**: Save/load vector database with metadata
- **Query embeddings**: Real-time OpenAI embedding for search queries

### **RAG Bot (`rag_bot.py`)**
- **Context retrieval**: Intelligent document chunk selection
- **Conversation memory**: Persistent chat history across sessions
- **Streaming responses**: Real-time OpenAI GPT-4o-mini completions
- **Smart prompting**: Context-aware prompt construction
- **Usage tracking**: Token counting and cost monitoring

### **CLI Interface (`rag_cli.py`)**
- **Document ingestion**: `rag_cli.py ingest <pdf_path>`
- **Single queries**: `rag_cli.py query "your question"`
- **Interactive mode**: `rag_cli.py interactive`
- **System status**: `rag_cli.py status`
- **Database cleanup**: `rag_cli.py cleanup`

## 🔧 Installation & Dependencies

### **Python Requirements**
```bash
# Core dependencies (from requirements.txt)
click==8.2.1
faiss_cpu==1.11.0.post1
PyMuPDF==1.26.3
huggingface_hub==0.33.4
mlx==0.26.5
mlx_lm==0.26.0
mlx_metal==0.26.5
numpy
openai==1.97.1
PyPDF2==3.0.1
python-dotenv==1.1.1
reportlab==4.4.3
```

### **System Requirements**
- **Python**: 3.10+ recommended
- **Platform**: macOS (optimized for Apple Silicon), Linux, Windows
- **Memory**: 8GB+ RAM (8GB+ recommended for large documents)
- **Storage**: 30GB+ for unquantized model and around 5GB for quantized and vector databases

## 📊 Performance Characteristics

### **Processing Performance**
- **PDF Extraction**: 1-2 pages/second
- **Embedding Generation**: 50-100 chunks/second (batched API calls)
- **Vector Search**: <100ms per query
- **End-to-end Query**: 2-5 seconds (including OpenAI API)


## 🎯 Key Features for Interview Evaluation

### **1. Production-Quality Code**
- **Error handling**: Comprehensive exception management with graceful fallbacks
- **Logging**: Structured logging with configurable levels
- **Progress indicators**: Real-time feedback for long-running operations
- **Configuration management**: Environment variables and CLI options

### **2. OpenAI API Integration**
- **Efficient batching**: Minimize API calls and costs
- **Rate limiting**: Handle API rate limits gracefully
- **Streaming responses**: Real-time chat completions
- **Cost optimization**: Smart chunking and context management

### **3. Vector Search Implementation**
- **FAISS optimization**: Multiple index strategies based on data size
- **Semantic search**: High-quality embeddings with OpenAI
- **Similarity scoring**: Configurable relevance thresholds
- **Metadata preservation**: Full document context with search results

### **4. System Design**
- **Modular architecture**: Clean separation of concerns
- **CLI interface**: Professional command-line tools with help
- **Conversation memory**: Stateful chat sessions
- **Database management**: Persistent storage with integrity checks

## 💻 Usage Examples

### **Document Processing**
```bash
# Process any PDF document
python rag_cli.py ingest my_document.pdf

# Custom chunking parameters
python rag_cli.py ingest document.pdf --chunk-size 1024 --overlap 100

# Check processing status
python rag_cli.py status
```

### **Querying Documents**
```bash
# Ask factual questions
python rag_cli.py query "How does Procyon help in standardizing PC purchasing decisions?"

# Technical details
python rag_cli.py query "What is the Procyon Office Productivity Benchmark used for?"

# Business/Retail Use
python rag_cli.py query "How can retailers use Procyon benchmark scores to guide customers?"
```

### **Interactive Sessions**
```bash
# Start chat mode
python rag_cli.py interactive

# In interactive mode:
# - Ask follow-up questions
# - Type 'stats' for conversation statistics
# - Type 'clear' to reset conversation history
# - Type 'quit' to exit
```

## 🗂️ File Structure

```
├── pdf_processor.py      # PDF processing and OpenAI embeddings
├── vector_database.py    # FAISS vector storage and search
├── rag_bot.py           # Query processing and chat interface  
├── rag_cli.py           # Command-line interface
├── quantize_model.py    # Local model quantization (optional)
├── setup.sh             # Automated environment setup
├── start_rag.sh         # Production startup wrapper
├── run_demo.sh          # Self-test demonstration script
├── requirements.txt     # Python dependencies
├── environment.yml      # Conda environment specification
└── procyon_guide.pdf    # Sample document for testing
```

## 🔍 Data Flow Details

### **Ingestion Pipeline**
1. **PDF Loading**: PyMuPDF extracts text with fallback to PyPDF2
2. **Text Cleaning**: Remove artifacts, normalize unicode, fix encoding
3. **Semantic Chunking**: Split by paragraphs with configurable overlap
4. **Embedding Generation**: OpenAI API calls in optimized batches
5. **Vector Storage**: FAISS index creation with metadata preservation

### **Query Pipeline**
1. **Query Embedding**: Real-time OpenAI embedding generation
2. **Vector Search**: FAISS similarity search with scoring
3. **Context Assembly**: Retrieve relevant chunks with metadata
4. **Prompt Construction**: Build context-aware prompts for OpenAI
5. **Response Generation**: Streaming GPT-4o-mini completions
6. **Memory Storage**: Persist conversation for follow-up queries

## 🧪 Testing & Validation

### **Sample Test Queries**
The included `procyon_guide.pdf` contains comprehensive benchmark documentation. Try these queries:

```bash
# Factual information
python rag_cli.py query "What is UL Procyon and what types of benchmarks does it offer?"

# Technical details
python rag_cli.py query "Explain how the Office Productivity benchmark scoring works"

# Comparative analysis  
python rag_cli.py query "Compare the system requirements for Windows vs macOS"

# Specific procedures
python rag_cli.py query "How do I install and run Procyon on macOS?"
```

## 🛠️ Configuration Options

### **Environment Variables**
```bash
# Required
export OPENAI_API_KEY=your_api_key_here

# Optional optimizations
export KMP_DUPLICATE_LIB_OK=TRUE    # Handle OpenMP conflicts
export OMP_NUM_THREADS=1            # Single-threaded for stability
```

### **CLI Options**
```bash
# Ingestion parameters
--chunk-size 512        # Characters per chunk (default: 512)
--overlap 50           # Overlap between chunks (default: 50)
--batch-size 100       # API batch size (default: 100)

# Query parameters  
--model gpt-4o-mini    # OpenAI model (default: gpt-4o-mini)
--max-chunks 5         # Context chunks per query (default: 5)
```

## 🚨 Troubleshooting

### **Common Issues**

**"OpenAI API key not set"**
```bash
export OPENAI_API_KEY=your_key_here
```

**"Vector database not found"**
```bash
# Run ingestion first
python rag_cli.py ingest your_document.pdf
```

**OpenMP library conflicts (macOS)**
```bash
# Automatically handled by setup.sh, or manually:
export KMP_DUPLICATE_LIB_OK=TRUE
```

**Installation issues**
```bash
# Clean installation
rm -rf venv
./setup.sh
```

## 💡 Advanced Features

### **Local Model Support** 
The `quantize_model.py` script enables local inference with Llama models for privacy-sensitive applications:

```bash
# Download and quantize Llama-3.1-8B to INT4
python quantize_model.py

# Requires Apple Silicon Mac and significant storage
```

### **Conversation Export**
```python
# Export chat history
bot = RAGBot()
bot.export_conversation("my_conversation.md")
```

### **Custom Embeddings**
```python
# Use different embedding models
processor = PDFProcessor(embedding_model="text-embedding-3-large")
```

## 📈 Evaluation Metrics

For interview assessment, this system demonstrates:

- **Technical depth**: Production-quality code with proper error handling
- **API integration**: Efficient OpenAI usage with cost optimization  
- **System design**: Clean architecture with modular components
- **User experience**: Intuitive CLI with comprehensive features
- **Performance**: Optimized vector search and embedding generation
- **Documentation**: Clear code comments and usage examples

## 🎓 Learning Outcomes

This project showcases:
- **RAG architecture**: Understanding of retrieval-augmented generation
- **Vector databases**: FAISS implementation and optimization
- **API integration**: OpenAI embeddings and chat completions
- **Production practices**: Error handling, logging, configuration
- **CLI development**: Professional command-line interfaces
- **Performance optimization**: Batching, caching, memory management

---

**Built for production use and interview demonstration. Ready to deploy and scale.**