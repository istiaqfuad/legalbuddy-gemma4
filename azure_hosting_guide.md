# Hosting Law Buddy (Gemma-2-27B + LoRA) on Azure

Hosting a 27B parameter model requires an environment with sufficient GPU memory. In 16-bit precision, the model requires roughly **54GB of VRAM** for weights alone, plus memory for the KV cache during generation. 

The ideal Azure VM for this is one with an **NVIDIA A100 (80GB)**, such as the `Standard_NC24ads_A100_v4`.

There are two primary ways to deploy this on Azure:

---

## Method 1: Azure Machine Learning (AML) Managed Online Endpoints (Recommended)
Azure ML provides managed endpoints that handle load balancing, scaling, and container management for you.

### Step 1: Merge the LoRA Adapter (Recommended)
While some serving frameworks can load LoRA adapters dynamically, it is much easier and more robust to merge the adapter into the base model first. You can do this using `peft` and `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "unsloth/gemma-2-27b-it"
lora_path = "./lawbuddy-prod-27b-16bit-final"
merged_path = "./lawbuddy-27b-merged"

# 1. Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 2. Apply LoRA and merge
print("Merging LoRA...")
model = PeftModel.from_pretrained(base_model, lora_path)
merged_model = model.merge_and_unload()

# 3. Save
print("Saving merged model...")
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
```

### Step 2: Upload to Azure ML Model Registry
Using the Azure CLI or Azure ML Python SDK, register the `lawbuddy-27b-merged` folder as a Model in your Azure ML Workspace.

### Step 3: Deploy using vLLM
**vLLM** is the industry standard for fast LLM inference. Azure ML allows you to bring your own container.

1. **Environment:** Create an Azure ML Environment using the official vLLM Docker image (`vllm/vllm-openai:latest`).
2. **Endpoint:** Create a Managed Online Endpoint.
3. **Deployment:** Create a deployment under that endpoint. 
   - **Compute:** Select `Standard_NC24ads_A100_v4` (1x A100 80GB).
   - **Environment Variables:** Set `MODEL_ID` to the path where Azure mounts your registered model.
   - **Command:** Have the container run `python -m vllm.entrypoints.openai.api_server --model <path_to_model> --dtype bfloat16 --max-model-len 4096`.

This will expose an API that perfectly matches the OpenAI API format (e.g., `/v1/chat/completions`), allowing you to use standard OpenAI client libraries to talk to your model!

---

## Method 2: Azure Virtual Machine (IaaS)
If you want total control over the server and don't want the overhead of Azure ML, you can rent a raw GPU VM.

### Step 1: Provision the VM
1. Go to the Azure Portal and create a new Virtual Machine.
2. Choose **Ubuntu Server 22.04 LTS (Data Science Virtual Machine)** (this comes with NVIDIA drivers pre-installed).
3. For the size, search for **NC A100 v4-series** and select `Standard_NC24ads_A100_v4`.

### Step 2: Transfer and Merge
Upload your base model and the LoRA adapter to the VM. Run the Python merging script from Method 1 to create a single model folder.

### Step 3: Run the vLLM Docker Container
SSH into your VM and run the vLLM server via Docker. This command maps port 8000 on the VM to the vLLM server inside the container:

```bash
docker run --gpus all \
    -v /path/to/your/lawbuddy-27b-merged:/model \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model /model \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```

### Step 4: Call Your API
Your model is now accessible via standard HTTP requests:

```bash
curl http://<YOUR_VM_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "messages": [
      {"role": "user", "content": "What is the penalty under Section 302?"}
    ],
    "temperature": 0.2
  }'
```

---

## Cost Optimization (4-bit / 8-bit)
If an 80GB A100 is too expensive (they run ~$3–$4/hour depending on your Azure region/agreement), you can quantize the merged model to run on much cheaper hardware.

If you load the model using **AWQ** or **bitsandbytes 4-bit quantization**, the memory footprint drops from 54GB down to about **16GB**. This allows you to host the model on a vastly cheaper VM, such as the `Standard_NC4as_T4_v3` (1x 16GB T4 GPU) or an L4 instance, cutting your hosting costs by up to 80%. vLLM supports serving AWQ and bitsandbytes models natively.
