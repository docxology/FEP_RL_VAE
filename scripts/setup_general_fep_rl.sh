#!/bin/bash
# Setup script to install general_FEP_RL dependency

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SIBLING_DIR="$REPO_ROOT/../active-inference-sim-lab"

echo "Setting up general_FEP_RL dependency..."

if [ ! -d "$SIBLING_DIR" ]; then
    echo "Error: active-inference-sim-lab directory not found at $SIBLING_DIR"
    echo "Please ensure the active-inference-sim-lab repository is cloned as a sibling directory."
    exit 1
fi

cd "$SIBLING_DIR"

if [ -f "pyproject.toml" ] || [ -f "setup.py" ]; then
    echo "Installing general_FEP_RL from $SIBLING_DIR..."
    if command -v uv &> /dev/null; then
        uv pip install -e .
    else
        pip install -e .
    fi
    echo "✓ general_FEP_RL installed successfully"
else
    echo "Warning: No installation files found in $SIBLING_DIR"
    echo "Please check the active-inference-sim-lab repository structure."
    exit 1
fi

cd "$REPO_ROOT"
echo "Setup complete!"
