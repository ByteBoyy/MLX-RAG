import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

import openai
from openai import OpenAI

from pdf_processor import DocumentChunk
from vector_database import VectorDatabase
from dotenv import load_dotenv

load_dotenv()



logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    timestamp: str
    user_query: str
    retrieved_chunks: List[str]  # chunk IDs
    context_used: str
    response: str
    token_usage: Dict

class RAGBot:
    """RAG Bot with OpenAI integration and conversation memory"""
    
    def __init__(self,
                 vector_db_path: str = "./vector_db",
                 openai_model: str = "gpt-4o-mini",
                 max_context_chunks: int = 5,
                 max_context_length: int = 3000,
                 conversation_history_file: str = "./conversation_history.json"):
        """
        Initialize RAG Bot
        
        Args:
            vector_db_path: Path to vector database
            openai_model: OpenAI model to use
            max_context_chunks: Maximum chunks to retrieve
            max_context_length: Maximum context length in characters
            conversation_history_file: File to store conversation history
        """
        self.vector_db_path = vector_db_path
        self.openai_model = openai_model
        self.max_context_chunks = max_context_chunks
        self.max_context_length = max_context_length
        self.conversation_history_file = Path(conversation_history_file)
        
        # Initialize components
        self.vector_db = None
        self.openai_client = None
        self.conversation_history = []
        
        # Setup
        self._setup_openai()
        self._load_vector_database()
        self._load_conversation_history()
    
    def _setup_openai(self):
        """Setup OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Get your API key from: https://platform.openai.com/api-keys"
            )
        
        try:
            self.openai_client = OpenAI(api_key=api_key)
            logger.info(f"✓ OpenAI client initialized with model: {self.openai_model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
    
    def _load_vector_database(self):
        """Load the vector database"""
        try:
            self.vector_db = VectorDatabase(db_path=self.vector_db_path)
            if not self.vector_db.load():
                logger.warning("No existing vector database found")
            else:
                stats = self.vector_db.get_stats()
                logger.info(f"✓ Loaded vector database: {stats['total_chunks']} chunks")
        except Exception as e:
            logger.error(f"❌ Failed to load vector database: {e}")
            raise
    
    def _load_conversation_history(self):
        """Load conversation history from file"""
        if self.conversation_history_file.exists():
            try:
                with open(self.conversation_history_file, 'r') as f:
                    data = json.load(f)
                self.conversation_history = [
                    ConversationTurn(**turn) for turn in data
                ]
                logger.info(f"✓ Loaded {len(self.conversation_history)} conversation turns")
            except Exception as e:
                logger.warning(f"Failed to load conversation history: {e}")
                self.conversation_history = []
        else:
            self.conversation_history = []
    
    def _save_conversation_history(self):
        """Save conversation history to file"""
        try:
            data = [asdict(turn) for turn in self.conversation_history]
            with open(self.conversation_history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save conversation history: {e}")
    
    def retrieve_context(self, query: str) -> Tuple[List[DocumentChunk], str]:
        """
        Retrieve relevant context for the query
        
        Args:
            query: User query
            
        Returns:
            Tuple of (retrieved_chunks, formatted_context)
        """
        if not self.vector_db or self.vector_db.index is None:
            logger.warning("Vector database not available")
            return [], ""
        
        # Search for relevant chunks
        results = self.vector_db.search(
            query=query,
            k=self.max_context_chunks,
            score_threshold=0.1  # Minimum relevance threshold
        )
        
        if not results:
            logger.info("No relevant context found")
            return [], ""
        
        # Extract chunks and format context
        retrieved_chunks = [chunk for chunk, score in results]
        
        # Build context string with length limit
        context_parts = []
        total_length = 0
        
        for i, (chunk, score) in enumerate(results):
            chunk_text = f"[Context {i+1} - Page {chunk.page_number}]:\n{chunk.text}\n"
            
            if total_length + len(chunk_text) > self.max_context_length:
                break
            
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        formatted_context = "\n".join(context_parts)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks ({total_length} chars)")
        return retrieved_chunks, formatted_context
    
    def get_conversation_context(self, max_turns: int = 3) -> str:
        """
        Get recent conversation context
        
        Args:
            max_turns: Maximum recent turns to include
            
        Returns:
            Formatted conversation context
        """
        if not self.conversation_history:
            return ""
        
        recent_turns = self.conversation_history[-max_turns:]
        context_parts = []
        
        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_query}")
            context_parts.append(f"Assistant: {turn.response}")
        
        return "\n".join(context_parts)
    
    def build_prompt(self, query: str, document_context: str, conversation_context: str) -> str:
        """
        Build the complete prompt for OpenAI
        
        Args:
            query: User query
            document_context: Retrieved document context
            conversation_context: Recent conversation context
            
        Returns:
            Complete prompt string
        """
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided document context. 

Instructions:
1. Use the document context to answer the user's question accurately
2. If the context doesn't contain enough information to answer fully, say so
3. Always cite which context section(s) you're using in your answer
4. Be concise but comprehensive
5. If there's relevant conversation history, consider it for context but prioritize the document content
6. If the question cannot be answered from the provided context, clearly state this limitation

Document Context:
{document_context}

{conversation_section}

Please answer the following question based on the provided context:
{query}"""
        
        conversation_section = ""
        if conversation_context:
            conversation_section = f"\nRecent Conversation:\n{conversation_context}\n"
        
        return system_prompt.format(
            document_context=document_context or "[No relevant document context found]",
            conversation_section=conversation_section,
            query=query
        )
    
    def generate_response(self, prompt: str) -> Tuple[str, Dict]:
        """
        Generate response using OpenAI
        
        Args:
            prompt: Complete prompt to send to OpenAI
            
        Returns:
            Tuple of (response_text, token_usage)
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )
            
            # Handle streaming response
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print()  # New line after streaming
            
            # Note: With streaming, we don't get usage stats
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            return full_response, token_usage
            
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            return f"Sorry, I encountered an error: {e}", {}
    
    def query(self, user_input: str) -> str:
        """
        Process user query and return response
        
        Args:
            user_input: User's question/query
            
        Returns:
            Assistant's response
        """
        logger.info(f"🤔 Processing query: {user_input[:100]}...")
        
        # Retrieve relevant context
        retrieved_chunks, document_context = self.retrieve_context(user_input)
        
        # Get conversation context
        conversation_context = self.get_conversation_context()
        
        # Build prompt
        prompt = self.build_prompt(user_input, document_context, conversation_context)
        
        # Generate response
        print(f"\n🤖 Assistant: ", end="")
        response, token_usage = self.generate_response(prompt)
        
        # Save conversation turn
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_query=user_input,
            retrieved_chunks=[chunk.chunk_id for chunk in retrieved_chunks],
            context_used=document_context[:200] + "..." if len(document_context) > 200 else document_context,
            response=response,
            token_usage=token_usage
        )
        
        self.conversation_history.append(turn)
        self._save_conversation_history()
        
        return response
    
    def get_conversation_stats(self) -> Dict:
        """Get conversation statistics"""
        if not self.conversation_history:
            return {"total_turns": 0}
        
        total_tokens = sum(
            turn.token_usage.get("total_tokens", 0) 
            for turn in self.conversation_history
        )
        
        return {
            "total_turns": len(self.conversation_history),
            "total_tokens_used": total_tokens,
            "average_chunks_per_query": sum(
                len(turn.retrieved_chunks) for turn in self.conversation_history
            ) / len(self.conversation_history),
            "first_query": self.conversation_history[0].timestamp,
            "last_query": self.conversation_history[-1].timestamp
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        if self.conversation_history_file.exists():
            self.conversation_history_file.unlink()
        logger.info("✓ Conversation history cleared")
    
    def export_conversation(self, output_file: str):
        """Export conversation to a readable format"""
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            f.write("# RAG Bot Conversation Export\n\n")
            
            for i, turn in enumerate(self.conversation_history, 1):
                f.write(f"## Turn {i} - {turn.timestamp}\n\n")
                f.write(f"**User:** {turn.user_query}\n\n")
                f.write(f"**Assistant:** {turn.response}\n\n")
                f.write(f"**Chunks Used:** {len(turn.retrieved_chunks)}\n\n")
                f.write("---\n\n")
        
        logger.info(f"✓ Conversation exported to {output_path}")

def main():
    """Test the RAG bot"""
    import sys
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY=your_api_key_here")
        print("   Get your key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    try:
        # Initialize RAG bot
        bot = RAGBot()
        
        print("🤖 RAG Bot initialized successfully!")
        print("📚 Vector database loaded")
        print("💬 Ready to answer questions about your document")
        print("🔧 Type 'quit' to exit, 'stats' for statistics, 'clear' to clear history")
        print("=" * 60)
        
        # Interactive loop
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'stats':
                    stats = bot.get_conversation_stats()
                    print(f"\n📊 Conversation Statistics:")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                    
                    db_stats = bot.vector_db.get_stats()
                    print(f"\n📚 Database Statistics:")
                    for key, value in db_stats.items():
                        print(f"   {key}: {value}")
                    continue
                
                elif user_input.lower() == 'clear':
                    bot.clear_history()
                    print("✓ Conversation history cleared")
                    continue
                
                # Process query
                response = bot.query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"❌ Sorry, I encountered an error: {e}")
    
    except Exception as e:
        logger.error(f"Failed to initialize RAG bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()