#!/bin/bash
set -e

echo "Formatting and mounting data disk..."
sudo parted /dev/sdc --script mklabel gpt mkpart xfspart xfs 0% 100%
sudo mkfs.xfs /dev/sdc1
sudo mkdir -p /data
sudo mount /dev/sdc1 /data
sudo chown -R azureuser:azureuser /data

echo "Setting up 64GB Swap on data disk..."
sudo fallocate -l 64G /data/swapfile
sudo chmod 600 /data/swapfile
sudo mkswap /data/swapfile
sudo swapon /data/swapfile

cd /data

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
rm -rf lawbuddy-gemma4-31b-merged lawbuddy-merged.tar lawbuddy-f16.gguf azcopy.tar.gz llama.zip

echo "Starting Server!"
nohup ./llama-server -m lawbuddy-q4.gguf --host 0.0.0.0 --port 8080 -c 4096 -n 512 > server.log 2>&1 &
echo "Done! Server is running."
