#!/bin/bash
echo "Waiting for dpkg lock..."
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 5; done

echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y curl unzip wget

echo "Downloading llama-server binary..."
wget -qO llama.zip https://github.com/ggerganov/llama.cpp/releases/download/b3492/llama-b3492-bin-ubuntu-x64.zip
unzip -jo llama.zip build/bin/llama-server
chmod +x llama-server

echo "Installing azcopy..."
wget -qO azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
tar -xf azcopy.tar.gz
sudo cp ./azcopy_linux_amd64_*/azcopy /usr/local/bin/azcopy
sudo chmod +x /usr/local/bin/azcopy

echo "Waiting for lawbuddy-q4.gguf to appear in Azure Blob Storage..."
BLOB_URL="https://lawbuddy27bmodelfiles.blob.core.windows.net/models/lawbuddy-q4.gguf?<SAS_TOKEN>"
while ! azcopy copy "$BLOB_URL" "./lawbuddy-q4.gguf"; do
    echo "Model not ready in Blob Storage yet (Molab is still converting). Retrying in 30 seconds..."
    sleep 30
done

echo "Model downloaded successfully! Starting llama-server..."
nohup ./llama-server -m lawbuddy-q4.gguf --host 0.0.0.0 --port 8080 -c 4096 -n 512 > server.log 2>&1 &
echo "Server is live on port 8080!"
