#!/bin/bash
/home/johnj/comfy_dev_mcp_server/python-mcp-server/reset_gpu.sh
cd /home/johnj/ComfyUI
source .venv/bin/activate

# PyIsolate: Use legacy allocator for MODEL isolation support
# Feature flag: Set PYISOLATE_DISABLE_CUDAMALLOCASYNC=0 to use cudaMallocAsync
if [ "${PYISOLATE_DISABLE_CUDAMALLOCASYNC:-1}" = "1" ]; then
    # Use ONLY modern env var to avoid conflicts
    export PYTORCH_ALLOC_CONF="backend:native"
    echo "📚 [PyIsolate] Using legacy CUDA allocator (backend:native)"
else
    export PYTORCH_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:False"
    echo "📚 [PyIsolate] Using cudaMallocAsync allocator"
fi

python main.py --listen --enable-cors-header "*"
