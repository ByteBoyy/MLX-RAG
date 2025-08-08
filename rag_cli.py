import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import click

from pdf_processor import PDFProcessor
from vector_database import VectorDatabase
from rag_bot import RAGBot
from dotenv import load_dotenv

load_dotenv()



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--db-path', default='./vector_db', help='Vector database path')
@click.pass_context
def cli(ctx, verbose, db_path):
    """RAG CLI - Retrieval-Augmented Generation System with OpenAI Embeddings"""
    ctx.ensure_object(dict)
    ctx.obj['db_path'] = db_path
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--chunk-size', default=512, help='Maximum chunk size in characters')
@click.option('--overlap', default=50, help='Overlap between chunks')
@click.option('--batch-size', default=100, help='Batch size for OpenAI API calls')
@click.option('--force', is_flag=True, help='Force reprocessing even if database exists')
@click.pass_context
def ingest(ctx, pdf_path, chunk_size, overlap, batch_size, force):
    """Ingest PDF document into vector database"""
    db_path = ctx.obj['db_path']
    
    click.echo(f"📄 Ingesting PDF: {pdf_path}")
    click.echo(f"🗂️  Database path: {db_path}")
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        click.echo("❌ OPENAI_API_KEY environment variable not set", err=True)
        click.echo("💡 Get your API key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    try:
        # Check if database already exists
        db_dir = Path(db_path)
        if db_dir.exists() and not force:
            if click.confirm("Vector database already exists. Overwrite?"):
                click.echo("🗑️  Removing existing database...")
                import shutil
                shutil.rmtree(db_dir)
            else:
                click.echo("❌ Ingestion cancelled")
                return
        
        # Process PDF
        click.echo("🔄 Processing PDF...")
        processor = PDFProcessor()
        
        # Extract and chunk text (this part is fast)
        click.echo("📄 Extracting text from PDF...")
        pages = processor.load_pdf(pdf_path)
        
        click.echo("✂️  Creating text chunks...")
        chunks = processor.create_chunks(pages, chunk_size, overlap)
        
        click.echo(f"📊 Created {len(chunks)} chunks from {len(pages)} pages")
        
        # Generate embeddings with progress indication
        click.echo("🧠 Generating embeddings using OpenAI API...")
        click.echo(f"   Model: text-embedding-3-small")
        click.echo(f"   Batch size: {batch_size}")
        click.echo(f"   Approximate cost: ${len(chunks) * 50 / 1000000 * 0.02:.4f}")
        
        # Show progress for embedding generation
        num_batches = (len(chunks) + batch_size - 1) // batch_size
        
        with click.progressbar(
            length=num_batches, 
            label="Generating embeddings",
            show_eta=True
        ) as bar:
            # Custom embedding generation with progress updates
            chunks_with_embeddings = processor.generate_embeddings_with_progress(
                chunks, batch_size, progress_callback=lambda: bar.update(1)
            )
        
        # Create vector database
        click.echo("🔍 Creating vector database...")
        vector_db = VectorDatabase(db_path=db_path)
        
        click.echo("📇 Building search index...")
        vector_db.add_chunks(chunks_with_embeddings)
        
        # Save database
        click.echo("💾 Saving database...")
        vector_db.save()
        
        # Print statistics
        stats = vector_db.get_stats()
        click.echo("\n📊 Ingestion Summary:")
        click.echo(f"   📄 Total chunks: {stats['total_chunks']}")
        click.echo(f"   📖 Pages covered: {stats['total_pages']}")
        click.echo(f"   📏 Average chunk length: {stats['avg_chunk_length']:.1f} chars")
        click.echo(f"   🧠 Embedding model: {stats['embedding_model']}")
        click.echo(f"   📐 Embedding dimension: {stats['dimension']}")
        
        click.echo(f"\n✅ PDF successfully ingested into {db_path}")
        click.echo("💡 You can now query the document using: rag_cli query \"your question\"")
        
    except Exception as e:
        click.echo(f"❌ Ingestion failed: {e}", err=True)
        logger.exception("Detailed error information:")
        sys.exit(1)

@cli.command()
@click.argument('query_text')
@click.option('--model', default='gpt-4o-mini', help='OpenAI model to use')
@click.option('--max-chunks', default=5, help='Maximum context chunks to retrieve')
@click.option('--interactive', '-i', is_flag=True, help='Start interactive session after query')
@click.pass_context
def query(ctx, query_text, model, max_chunks, interactive):
    """Query the document with a question"""
    db_path = ctx.obj['db_path']
    
    # Check if database exists
    if not Path(db_path).exists():
        click.echo("❌ Vector database not found. Please run 'ingest' command first.", err=True)
        sys.exit(1)
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        click.echo("❌ OPENAI_API_KEY environment variable not set", err=True)
        click.echo("💡 Get your API key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    try:
        # Initialize RAG bot
        click.echo("🤖 Initializing RAG bot...")
        bot = RAGBot(
            vector_db_path=db_path,
            openai_model=model,
            max_context_chunks=max_chunks
        )
        
        # Process query
        click.echo(f"🔍 Query: {query_text}")
        click.echo("=" * 60)
        
        response = bot.query(query_text)
        
        click.echo("\n" + "=" * 60)
        
        # Start interactive session if requested
        if interactive:
            click.echo("\n💬 Starting interactive session...")
            click.echo("🔧 Type 'quit' to exit, 'stats' for statistics, 'clear' to clear history")
            
            while True:
                try:
                    user_input = click.prompt("\n👤 You", type=str).strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    elif user_input.lower() == 'stats':
                        stats = bot.get_conversation_stats()
                        click.echo("\n📊 Conversation Statistics:")
                        for key, value in stats.items():
                            click.echo(f"   {key}: {value}")
                        continue
                    
                    elif user_input.lower() == 'clear':
                        bot.clear_history()
                        click.echo("✓ Conversation history cleared")
                        continue
                    
                    # Process query
                    response = bot.query(user_input)
                    
                except KeyboardInterrupt:
                    break
            
            click.echo("\n👋 Goodbye!")
        
    except Exception as e:
        click.echo(f"❌ Query failed: {e}", err=True)
        logger.exception("Detailed error information:")
        sys.exit(1)

@cli.command()
@click.pass_context
def interactive(ctx):
    """Start interactive chat session"""
    db_path = ctx.obj['db_path']
    
    # Check if database exists
    if not Path(db_path).exists():
        click.echo("❌ Vector database not found. Please run 'ingest' command first.", err=True)
        sys.exit(1)
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        click.echo("❌ OPENAI_API_KEY environment variable not set", err=True)
        click.echo("💡 Get your API key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    try:
        # Initialize RAG bot
        click.echo("🤖 Initializing RAG bot...")
        bot = RAGBot(vector_db_path=db_path)
        
        click.echo("\n🎉 RAG Bot ready!")
        click.echo("📚 Vector database loaded")
        click.echo("💬 Ask questions about your document")
        click.echo("🔧 Commands: 'quit' to exit, 'stats' for statistics, 'clear' to clear history")
        click.echo("=" * 60)
        
        while True:
            try:
                user_input = click.prompt("\n👤 You", type=str).strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                elif user_input.lower() == 'stats':
                    stats = bot.get_conversation_stats()
                    click.echo("\n📊 Conversation Statistics:")
                    for key, value in stats.items():
                        click.echo(f"   {key}: {value}")
                    
                    db_stats = bot.vector_db.get_stats()
                    click.echo("\n📚 Database Statistics:")
                    for key, value in db_stats.items():
                        click.echo(f"   {key}: {value}")
                    continue
                
                elif user_input.lower() == 'clear':
                    bot.clear_history()
                    click.echo("✓ Conversation history cleared")
                    continue
                
                # Process query
                response = bot.query(user_input)
                
            except KeyboardInterrupt:
                break
        
        click.echo("\n👋 Goodbye!")
        
    except Exception as e:
        click.echo(f"❌ Interactive session failed: {e}", err=True)
        logger.exception("Detailed error information:")
        sys.exit(1)

@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and statistics"""
    db_path = ctx.obj['db_path']
    
    click.echo("🔍 RAG System Status")
    click.echo("=" * 40)
    
    # Check database
    db_dir = Path(db_path)
    if db_dir.exists():
        try:
            vector_db = VectorDatabase(db_path=db_path)
            vector_db.load()
            stats = vector_db.get_stats()
            
            click.echo("📚 Vector Database: ✅ Ready")
            click.echo(f"   📍 Path: {db_path}")
            click.echo(f"   📄 Chunks: {stats['total_chunks']}")
            click.echo(f"   📖 Pages: {stats['total_pages']}")
            click.echo(f"   🧠 Model: {stats['embedding_model']}")
            click.echo(f"   📐 Dimension: {stats['dimension']}")
            click.echo(f"   📏 Avg chunk size: {stats['avg_chunk_length']:.1f} chars")
            
        except Exception as e:
            click.echo("📚 Vector Database: ❌ Error")
            click.echo(f"   Error: {e}")
    else:
        click.echo("📚 Vector Database: ❌ Not found")
        click.echo("   💡 Run 'rag_cli ingest <pdf_path>' to create database")
    
    # Check OpenAI API key
    if os.getenv('OPENAI_API_KEY'):
        click.echo("🤖 OpenAI API: ✅ Key found")
        click.echo("   🧠 Embeddings: text-embedding-3-small")
        click.echo("   💬 Chat: gpt-4o-mini (default)")
    else:
        click.echo("🤖 OpenAI API: ❌ Key not set")
        click.echo("   💡 Set OPENAI_API_KEY environment variable")
    
    # Check conversation history
    history_file = Path("./conversation_history.json")
    if history_file.exists():
        try:
            import json
            with open(history_file, 'r') as f:
                history = json.load(f)
            click.echo(f"💬 Conversation History: ✅ {len(history)} turns")
        except:
            click.echo("💬 Conversation History: ❌ Corrupted")
    else:
        click.echo("💬 Conversation History: 📝 Empty")

@cli.command()
@click.pass_context
def cleanup(ctx):
    """Clean up database and conversation history"""
    db_path = ctx.obj['db_path']
    
    if click.confirm("⚠️  This will delete the vector database and conversation history. Continue?"):
        # Remove database
        db_dir = Path(db_path)
        if db_dir.exists():
            import shutil
            shutil.rmtree(db_dir)
            click.echo(f"✅ Removed vector database: {db_path}")
        
        # Remove conversation history
        history_file = Path("./conversation_history.json")
        if history_file.exists():
            history_file.unlink()
            click.echo("✅ Removed conversation history")
        
        click.echo("🧹 Cleanup complete")
    else:
        click.echo("❌ Cleanup cancelled")

if __name__ == '__main__':
    # Handle direct script execution with arguments
    if len(sys.argv) > 1:
        # Use click CLI
        cli()
    else:
        # Show help if no arguments
        click.echo("RAG CLI - Retrieval-Augmented Generation System with OpenAI Embeddings")
        click.echo("\nUsage examples:")
        click.echo("  python rag_cli.py ingest document.pdf")
        click.echo("  python rag_cli.py query \"What is the main topic?\"")
        click.echo("  python rag_cli.py interactive")
        click.echo("  python rag_cli.py status")
        click.echo("\nFor full help: python rag_cli.py --help")