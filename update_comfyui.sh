#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

COMFYUI_DIR="/home/johnj/ComfyUI"
VENV_PYTHON="$COMFYUI_DIR/.venv/bin/python"
VENV_PIP="$COMFYUI_DIR/.venv/bin/pip"

echo "--- ComfyUI Update Script Started ---"

echo "[INFO] Looking for existing ComfyUI/GPU processes via nvidia-smi..."
# User's command to kill GPU processes.
# Added '|| true' after grep so script doesn't exit if no PIDs are found.
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eo '[0-9]+' || true)

if [ -n "$PIDS" ]; then
    echo "[INFO] Found GPU PIDs: $PIDS. Attempting to kill -9..."
    kill -9 $PIDS || echo "[WARN] Failed to kill some PIDs (they might have already exited)."
else
    echo "[INFO] No active GPU PIDs found by nvidia-smi that match."
fi
# A brief pause to ensure processes are terminated before restart attempts by systemd (if any other auto-restart is configured)
sleep 2

cd "$COMFYUI_DIR" || { echo "[ERROR] Failed to cd to $COMFYUI_DIR"; exit 1; }
echo "[INFO] Changed directory to $COMFYUI_DIR"

echo "[INFO] Activating virtual environment for pip (by direct path)..."
# Using pip from venv directly

echo "[INFO] Updating PyTorch (nightly cu129)..."
"$VENV_PIP" install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129

echo "[INFO] PyTorch update command finished."

echo "[INFO] Pulling latest changes for ComfyUI via git..."
git pull
echo "[INFO] Git pull finished."

echo "[INFO] Updating custom nodes using update_nodes.sh..."
if [ -f "./update_nodes.sh" ]; then
    # Ensure update_nodes.sh is executable if it isn't already
    chmod +x ./update_nodes.sh
    ./update_nodes.sh "/home/johnj/ComfyUI/custom_nodes"
    echo "[INFO] update_nodes.sh script finished."
else
    echo "[WARN] ./update_nodes.sh not found, skipping custom node update."
fi

echo "[INFO] Installing/updating Python requirements from requirements.txt..."
"$VENV_PIP" install -r requirements.txt
echo "[INFO] Requirements installation finished."

echo "[INFO] Requesting systemd to restart ComfyUI service..."
sudo systemctl restart comfyui.service
echo "[INFO] ComfyUI service restart requested."

echo "--- ComfyUI Update Script Finished ---"
