# Azure infrastructure — LegalBuddy model serving

One-off provisioning scripts used to convert the fine-tuned model to a
quantized GGUF and serve it on Azure for the competition demo. Kept for
reproducibility; not part of the application stack (that lives in
`legal-buddy/docker-compose.full.yml`).

## Pipeline

```
training/train_gemma4.py          (Molab, 96GB GPU)
        └─ lawbuddy-gemma4-31b-merged/     merged 16-bit checkpoint
                └─ vm_quantize_new.sh      HF → GGUF F16 → Q4_K_M
                        └─ lawbuddy-q4.gguf (~9 GB) → Azure Blob Storage
                                └─ setup_vm_new.sh   download + llama-server on VM
finish_fast.sh  = full on-host pipeline: extract → convert → quantize → serve
```

## Files

| file | purpose |
|---|---|
| `vm_quantize.sh` / `vm_quantize_new.sh` | Convert merged HF checkpoint to GGUF F16, quantize to Q4_K_M, upload |
| `setup_vm.sh` / `setup_vm_new.sh` | VM provisioning — download GGUF from Blob Storage, start llama-server |
| `finish_fast.sh` | Single-host pipeline: extract tarball → convert → quantize → serve |
| `azure_hosting_guide.md` | Handover notes for the Azure VM setup |
| `deployment.yml` / `endpoint.yml` | Azure ML managed online endpoint (vLLM) alternative |

## Notes

- SAS tokens in these scripts are redacted — generate fresh read-only SAS URLs
  before use.
- The GGUF artifact is served either this way (Azure VM) or locally via
  `legal-buddy/docker-compose.full.yml`; both use the same llama.cpp server
  with `--alias lawbuddy-gemma4`.
