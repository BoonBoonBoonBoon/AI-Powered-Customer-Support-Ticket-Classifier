# Windows GPU Setup for Training

This guide helps you run transformer training on a GPU in Windows. Choose the path that matches your hardware.

## 0) Check your hardware
- NVIDIA GPU: Preferred for PyTorch CUDA builds
- AMD/Intel GPU: You can use DirectML (via `torch-directml`)

Optional checks (PowerShell):
```powershell
# Show GPU adapters
Get-CimInstance Win32_VideoController | Select-Object Name
```

## 1) NVIDIA path (CUDA)
PyTorch publishes CUDA wheels for specific Python versions. As of now, CUDA wheels are best supported on Python 3.10–3.12.

Steps:
1) Install Python 3.12 (64‑bit) from https://www.python.org/downloads/windows/
2) Create a new venv for this repo with Python 3.12:
```powershell
py -3.12 -m venv .venv312
& ".\.venv312\Scripts\Activate.ps1"
```
3) Install CUDA‑enabled PyTorch (example: CUDA 12.1):
```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
4) Install project deps:
```powershell
python -m pip install -r requirements.txt
```
5) Verify CUDA is detected:
```powershell
python - << 'PY'
import torch
print({
  'torch': torch.__version__,
  'cuda_available': torch.cuda.is_available(),
  'cuda_version': getattr(torch.version, 'cuda', None),
})
PY
```
You should see `cuda_available: True` and a non‑null `cuda_version`.

## 2) AMD/Intel path (DirectML)
If you don’t have NVIDIA, you can still use your GPU via DirectML.

Steps:
1) Keep your current Python (3.10–3.13 ok) and venv
2) Install `torch-directml`:
```powershell
python -m pip install torch-directml
```
3) Use the new device flag in our trainer:
```powershell
& ".\.venv\Scripts\python.exe" scripts\train_transformer.py --data "data\enriched_customer_tickets.csv" --model-name distilroberta-base --epochs 5 --batch-size 16 --grad-accum 2 --lr 3e-5 --output-version t1.dml.test --device dml
```
The trainer will move the model and tensors to the DirectML device.

## 3) Auto device selection
We added `--device` to the trainer with options: `auto` (default), `cuda`, `cpu`, `dml`.
- `auto`: prefers CUDA, then DirectML if available, else CPU.
- `cuda`: fails fast if CUDA isn’t available.
- `dml`: uses DirectML (requires `torch-directml`).

## 4) Tips
- Keep batch size modest on GPU first; increase once it’s stable.
- If CUDA install fails, ensure Windows NVIDIA driver is up‑to‑date and you used a supported Python version.
- When switching venvs, remember to reinstall project requirements.
