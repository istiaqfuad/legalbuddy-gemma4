#!/bin/bash
set -e
echo "Setting up 64GB Swap..."
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

echo "Installing dependencies..."
sudo apt update
sudo apt install -y python3-pip wget unzip

echo "Downloading azcopy..."
wget -qO azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
tar -xf azcopy.tar.gz
sudo cp ./azcopy_linux_amd64_*/azcopy /usr/local/bin/azcopy
sudo chmod +x /usr/local/bin/azcopy

echo "Downloading merged model tar from Azure Blob Storage..."
BLOB_URL="https://lawbuddy27bmodelfiles.blob.core.windows.net/models/lawbuddy-gemma4-31b-merged.tar?<SAS_TOKEN>"
azcopy copy "$BLOB_URL" "./lawbuddy-merged.tar"

echo "Extracting tar..."
tar -xf lawbuddy-merged.tar

echo "Setting up llama.cpp tools..."
pip3 install --no-cache-dir gguf transformers sentencepiece protobuf numpy
wget -qO convert_hf_to_gguf.py https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py

wget -qO llama.zip https://github.com/ggerganov/llama.cpp/releases/download/b3492/llama-b3492-bin-ubuntu-x64.zip
unzip -jo llama.zip build/bin/llama-quantize build/bin/llama-server
chmod +x llama-quantize llama-server

echo "Converting to F16..."
python3 convert_hf_to_gguf.py lawbuddy-gemma4-31b-merged --outtype f16 --outfile lawbuddy-f16.gguf

echo "Quantizing to Q4_K_M..."
./llama-quantize lawbuddy-f16.gguf lawbuddy-q4.gguf Q4_K_M

echo "Cleaning up raw files to save disk space..."
rm -rf lawbuddy-gemma4-31b-merged lawbuddy-merged.tar lawbuddy-f16.gguf

echo "Starting Server!"
nohup ./llama-server -m lawbuddy-q4.gguf --host 0.0.0.0 --port 8080 -c 4096 -n 512 > server.log 2>&1 &
echo "Done! Server is running."
