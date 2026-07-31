# LawBuddy Fine-Tuning Project

Fine-tuned `google/gemma-4-31B-it` (16-bit LoRA) on legal case data, served via a 4-bit GGUF quantization, with a full RAG chat application.

## Repository Layout

```
├── training/          # Fine-tuning code & artifacts
│   ├── train_gemma4.py    # Production LoRA training (Gemma-4-31B, 16-bit)
│   ├── train_prod.py      # Earlier 27B training script
│   ├── style_sft_prod/    # SFT dataset (train/eval JSONL)
│   ├── dataset.tar.gz     # Packaged training dataset
│   └── train_prod.log*    # Training logs
├── legal-buddy/       # RAG chat application (FastAPI + Next.js)
├── deployment.yml     # Azure ML managed online deployment (vLLM)
├── endpoint.yml       # Azure ML managed online endpoint
├── setup_vm.sh        # VM provisioning — model download + llama.cpp serve
├── setup_vm_new.sh    # VM provisioning (fresh run)
├── vm_quantize.sh     # HF → GGUF conversion + Q4_K_M quantization
├── vm_quantize_new.sh # Quantization with data-disk setup
├── finish_fast.sh     # Final pipeline: extract, convert, quantize, serve
├── azure_hosting_guide.md
└── competition_description.txt
```

> **Note:** Model weights, raw legal PDFs (`legal_dataset/`, 1.9GB), and large tarballs are intentionally not committed (GitHub's 100MB/file limit). SAS tokens in the VM scripts are redacted — generate fresh read-only SAS URLs before use.

---

## What Has Been Done So Far

1.  **Infrastructure Setup**: 
    *   Successfully migrated to a high-spec Molab instance equipped with an RTX PRO 6000 (96GB VRAM) to support the 27B model.
    *   Configured the environment with `unsloth`, `trl`, and `xformers`, resolving complex dependency conflicts (like `torchvision` version mismatches) and API deprecations (like `dataset_text_field` in `trl`).

2.  **Model Training**:
    *   Fine-tuned the `unsloth/gemma-2-27b-it` model.
    *   Used full 16-bit precision (`bfloat16`/`float16`) to avoid the quantization degradation seen in previous 4-bit runs.
    *   Trained using LoRA adapters on a clean dataset of 47 high-quality examples for 60 steps. The training loss smoothly converged to ~0.49, indicating excellent adaptation without catastrophic overfitting.

3.  **Artifact Generation**:
    *   Successfully exported the trained LoRA adapter.
    *   Packaged the adapter, training code (`train_27b_16bit.py`), training logs, and the dataset into a downloadable archive (`lawbuddy_27b_16bit_results.zip`).

4.  **Inference Testing**:
    *   Ran a test prompt against the newly trained model.
    *   **Result**: The model successfully adopted the target formatting (e.g., `Judicial Reasoning:`, `Holding:`, `Clarifying Questions:`) and matched the authoritative legal tone perfectly. 
    *   **Identified Issues**: Due to the extremely small dataset size (47 examples), the model hallucinated some case facts (confusing the 15th and 19th Amendments) and got stuck in a generative loop asking clarifying questions at the end.

---

## What Needs to be Done for Production-Readiness

To fix the hallucinations and looping issues, and to make the model reliable enough for production, the following steps are required:

1.  **Dataset Expansion (Crucial)**
    *   **Current State**: 47 examples is excellent for stylistic transfer but insufficient for factual grounding and robust generalization.
    *   **Action**: Process the raw PDFs located in `legal_dataset/cases`. We need to use the exact same high-quality extraction pipeline (used for the first 47) to convert these PDFs into structured JSONL format. 
    *   **Target**: Scale the dataset to at least 500–1,000 high-quality examples. The data quality must remain strictly pristine; we do not need synthetic augmentation if we have real cases, but we must extract them carefully.

2.  **Fixing the End-of-Sequence (EOS) Loop**
    *   **Current State**: The model doesn't know when to stop generating, leading to infinite bullet points.
    *   **Action**: Ensure that the `eos_token` (End of Sequence token) is properly appended to every example during training. We must also verify that inference scripts correctly pass the `eos_token_id` to the `generate()` function so the model stops when its task is complete.

3.  **Hyperparameter Optimization**
    *   **Training**: With a larger dataset, we will need to re-evaluate the number of epochs, learning rate, and LoRA rank (`r`) to ensure it learns the new data without forgetting the base knowledge.
    *   **Inference**: Tune generation parameters like `temperature` (lower for more factual answers), `top_p`, and `repetition_penalty` to minimize hallucinations and prevent looping.

4.  **Robust Evaluation Pipeline**
    *   **Action**: Establish a systematic way to test the model beyond a single manual prompt. We need a hold-out test set of cases to measure factual accuracy, legal reasoning validity, and stylistic adherence before considering the model "production-ready".
