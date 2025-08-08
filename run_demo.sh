#!/bin/bash

VENV_DIR="venv"

# Ensure the setup script is executable
chmod +x setup.sh

# Run setup to check/install dependencies
source ./setup.sh

# Activate the virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
fi

if [ -f .env ]; then
  echo "🔑 Loading environment variables from .env file..."
  while IFS='=' read -r key value; do
    # Skip lines starting with # or blank lines
    if [[ -n "$key" && ! "$key" =~ ^# ]]; then
      # Remove surrounding quotes if present
      value="${value%\"}"
      value="${value#\"}"
      export "$key"="$value"
    fi
  done < .env
fi

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY is not set. Please set it in your .env file (e.g., OPENAI_API_KEY=sk-...)"
    exit 1
fi

# Check if the document exists
if [ ! -f "procyon_guide.pdf" ]; then
    echo "❌ procyon_guide.pdf not found. Please ensure the file exists in the current directory."
    exit 1
fi

# Download and quantize the model
echo "🔄 Downloading and quantizing the model..."
python quantize_model.py

echo "🚀 Starting RAG CLI Demo..."
echo "================================================="

# Create a dummy PDF for the demo
echo "1. Ingesting the document..."
python rag_cli.py ingest procyon_guide.pdf --chunk-size 1024 --overlap 100

echo -e "\n2. Running a sample query..."
python rag_cli.py query "What is this document about?"

echo -e "\n3. Starting interactive session..."
echo "You can now ask questions about the document. Type 'quit' to exit."
echo "-------------------------------------------------"
python rag_cli.py interactive

echo "================================================="
echo "✅ Demo finished."
