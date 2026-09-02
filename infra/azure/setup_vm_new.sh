#!/bin/bash
echo "Killing previous setup processes..."
pkill -f azcopy || true
pkill -f llama-server || true

echo "Waiting for lawbuddy-q4.gguf to appear in Azure Blob Storage..."
BLOB_URL="https://lawbuddy27bmodelfiles.blob.core.windows.net/models/lawbuddy-q4.gguf?<SAS_TOKEN>"
while ! azcopy copy "$BLOB_URL" "./lawbuddy-q4.gguf"; do
    echo "Model not ready in Blob Storage yet (Molab is still converting). Retrying in 30 seconds..."
    sleep 30
done

echo "Model downloaded successfully! Starting llama-server..."
nohup ./llama-server -m lawbuddy-q4.gguf --host 0.0.0.0 --port 8080 -c 4096 -n 512 > server.log 2>&1 &
echo "Server is live on port 8080!"
