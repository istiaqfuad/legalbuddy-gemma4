#!/bin/bash
cd /data

echo "Extracting tar with 1M record size..."
tar --record-size=1048576 -xf lawbuddy-merged.tar

echo "Installing pip dependencies..."
pip3 install huggingface_hub hf_transfer transformers accelerate safetensors > pip_install.log 2>&1

echo "Converting to GGUF (F16)..."
python3 /data/convert_hf_to_gguf.py /data/lawbuddy-gemma4-31b-merged --outfile /data/lawbuddy.gguf --outtype f16 > convert.log 2>&1

echo "Quantizing to Q4_K_M..."
/data/llama.cpp/llama-quantize /data/lawbuddy.gguf /data/lawbuddy-q4.gguf Q4_K_M > quantize.log 2>&1

echo "Starting llama-server..."
nohup /data/llama.cpp/llama-server -m /data/lawbuddy-q4.gguf --host 0.0.0.0 --port 8080 -c 4096 > /data/server.log 2>&1 &
echo "Done!"
