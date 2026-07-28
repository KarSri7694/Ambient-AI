# ROCm Hackathon Setup

Use this profile for AMD Track 2 evidence without changing the production pipeline design.

1. Install Python 3.12 and the matching ROCm requirements.

   On Radeon Cloud Linux:

   ```bash
   python3 -m venv .venv-rocm
   source .venv-rocm/bin/activate
   python scripts/install_requirements.py --target rocm-linux
   ```

   On Windows ROCm:

   ```powershell
   py -3.12 -m venv .venv-rocm
   .\.venv-rocm\Scripts\python -m pip install --upgrade pip
   .\.venv-rocm\Scripts\python scripts\install_requirements.py --target rocm-windows
   ```

2. Install AMD's ROCm 7.2.1 validated `llama.cpp` Windows package, then copy `models_preset.rocm.example.ini` to `models_preset.ini` and replace the model paths.

3. Copy `config.example.ini` to `config.ini`, then apply the values from `config.rocm.example.ini`. Whisper remains the default ASR family. On Radeon Cloud, install with `scripts/install_requirements.py --target rocm-linux`; it installs the official CTranslate2 ROCm wheel first, then installs `faster-whisper` without replacing CTranslate2. Qwen ASR is optional and must be selected explicitly.

4. Run preflight:

   ```powershell
   .\.venv-rocm\Scripts\python scripts\rocm_preflight.py --require-supported-gpu --dependency-profile rocm-windows --asr-backend whisper_auto --llama-server D:\path\to\llama-server.exe
   ```

   On Radeon Cloud Linux, use:

   ```bash
   python scripts/rocm_preflight.py --require-supported-gpu --dependency-profile rocm-linux --asr-backend ctranslate2_rocm
   ```

   If a supported RDNA2 card needs the CTranslate2 allocator workaround, run Ambient with:

   ```bash
   export CT2_CUDA_ALLOCATOR=cub_caching
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
