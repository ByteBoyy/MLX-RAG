#!/bin/bash

# --- setup.sh ---
# This script checks for the required Python dependencies and sets up a 
# virtual environment if they are not met.

VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check for python3
check_python() {
    if command_exists python3; then
        echo "✓ Python 3 found."
    else
        echo "❌ Python 3 is not installed. Please install Python 3 to continue."
        exit 1
    fi
}

# Function to check for required dependencies
check_dependencies() {
    echo "Checking for dependencies..."
    
    # Check if pip is in a virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "✓ Already in a virtual environment."
        return 0
    fi
    
    # A simple check for some key packages.
    python3 -c "import click; import faiss; import fitz; import openai; import PyPDF2" &> /dev/null
    if [ $? -eq 0 ]; then
        echo "✓ Dependencies seem to be installed."
        return 0
    else
        echo "⚠️  Dependencies not found in the global environment."
        return 1
    fi
}

# Main setup logic
main() {
    check_python
    
    if check_dependencies; then
        echo "✅ Setup complete. Dependencies are satisfied."
    else
        echo "🔧 Setting up a virtual environment..."
        
        # Create virtual environment
        if [ ! -d "$VENV_DIR" ]; then
            python3 -m venv "$VENV_DIR"
            if [ $? -ne 0 ]; then
                echo "❌ Failed to create a virtual environment."
                exit 1
            fi
        fi
        
        # Activate virtual environment and install dependencies
        source "$VENV_DIR/bin/activate"
        
        echo "📦 Installing dependencies from $REQUIREMENTS_FILE..."
        pip install -r "$REQUIREMENTS_FILE"
        
        if [ $? -ne 0 ]; then
            echo "❌ Failed to install dependencies."
            exit 1
        fi
        
        echo "✅ Virtual environment created and dependencies installed successfully."
        echo "💡 To activate the environment manually, run: source $VENV_DIR/bin/activate"
    fi
}

main