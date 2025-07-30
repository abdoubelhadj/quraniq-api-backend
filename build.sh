#!/bin/bash

# Define Python and pip versions
PYTHON_VERSION="python3.10"
PIP_VERSION="pip3.10"

echo "--- Starting custom build script ---"
echo "Using Python version: $PYTHON_VERSION"

# Check if Python and pip binaries exist
if ! command -v "$PYTHON_VERSION" &> /dev/null; then
    echo "Error: Python version $PYTHON_VERSION not found. Please check available runtimes on Render."
    exit 1
fi

if ! command -v "$PIP_VERSION" &> /dev/null; then
    echo "Error: $PIP_VERSION not found. Please check available runtimes on Render."
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt with $PIP_VERSION..."
"$PIP_VERSION" install --disable-pip-version-check --target . --upgrade -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error installing Python dependencies."
    exit 1
fi

echo "Python dependencies installed successfully."

# Verify directory structure
echo "Verifying directory structure..."
if [ -d "api/app" ] && [ -f "api/app/chatbot.py" ]; then
    echo "Found api/app/chatbot.py"
else
    echo "Error: api/app/chatbot.py not found!"
    exit 1
fi

echo "Directory structure:"
ls -R

echo "--- Custom build script completed ---"
