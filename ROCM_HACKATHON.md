# ROCm Hackathon Setup

Use this profile for AMD Track 2 evidence without changing the production pipeline design.

1. Install Python 3.12 and the ROCm Windows requirements:

   ```powershell
   py -3.12 -m venv .venv-rocm
   .\.venv-rocm\Scripts\python -m pip install --upgrade pip
   .\.venv-rocm\Scripts\python -m pip install -r requirements-rocm-windows.txt
   ```

2. Install AMD's ROCm 7.2.1 validated `llama.cpp` Windows package, then copy `models_preset.rocm.example.ini` to `models_preset.ini` and replace the model paths.

3. Copy `config.example.ini` to `config.ini`, then apply the values from `config.rocm.example.ini`. For Windows ROCm, keep diarization, Demucs preprocessing, and speaker embedding on CPU until each library is validated on the target GPU.

4. Run preflight:

   ```powershell
   .\.venv-rocm\Scripts\python scripts\rocm_preflight.py --require-supported-gpu --llama-server D:\path\to\llama-server.exe
   ```

5. Start the AMD-tuned llama.cpp router:

   ```powershell
   scripts\start_rocm_llama.ps1 -LlamaServer D:\path\to\llama-server.exe -ModelsPreset .\models_preset.ini
   ```

6. Run benchmark evidence with one warmup and ten measured requests:

   ```powershell
   .\.venv-rocm\Scripts\python scripts\benchmark_rocm.py --model Ambient_Qwen25_VL_4B_ROCm --label tuned-rocm --output .ambient_data\benchmarks\rocm-tuned.json
   ```

7. Run real-world tests and capture the UI at `http://127.0.0.1:8766/real-world-tests`:

   ```powershell
   .\.venv-rocm\Scripts\python tests\real_world_tests\run_lab.py
   ```

For the hackathon submission, include the preflight JSON, benchmark JSON/CSV, real-world test export JSON/CSV, and a 3-5 minute demo showing GPU offload, VRAM usage, tool calls, and final loop results.
