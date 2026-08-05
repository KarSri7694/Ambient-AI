#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-faster-whisper-rocm}"
AUDIO_FILE="${AUDIO_FILE:-${SCRIPT_DIR}/audio.mp3}"
MODEL_NAME="${MODEL_NAME:-large-v3}"
LANGUAGE="${LANGUAGE:-}"
BEAM_SIZE="${BEAM_SIZE:-5}"
CTRANSLATE2_VERSION="${CTRANSLATE2_VERSION:-4.8.1}"
FASTER_WHISPER_VERSION="${FASTER_WHISPER_VERSION:-1.2.1}"

DOWNLOAD_DIR="${DOWNLOAD_DIR:-${ROOT_DIR}/.ambient_data/downloads/faster-whisper-rocm}"
FFMPEG_DIR="${FFMPEG_DIR:-${ROOT_DIR}/.ambient_data/tools/ffmpeg-linux64}"
CTRANSLATE2_ROCM_ZIP_URL="${CTRANSLATE2_ROCM_ZIP_URL:-https://github.com/OpenNMT/CTranslate2/releases/download/v${CTRANSLATE2_VERSION}/rocm-python-wheels-Linux.zip}"
FFMPEG_ZIP_URL="${FFMPEG_ZIP_URL:-https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.zip}"

log() {
  printf '\n[%s] %s\n' "$(date +'%H:%M:%S')" "$*"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

download_file() {
  local url="$1"
  local output="$2"
  if [[ -s "${output}" ]]; then
    log "Using cached $(basename "${output}")"
    return
  fi
  log "Downloading ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --output "${output}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${output}" "${url}"
  else
    die "Install curl or wget to download release archives."
  fi
}

ensure_linux() {
  [[ "$(uname -s)" == "Linux" ]] || die "This script installs Linux ROCm wheels and Linux FFmpeg builds. Run it on Linux."
  [[ "$(uname -m)" == "x86_64" ]] || die "This script expects x86_64 Linux."
}

ensure_rocm_visible() {
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi || true
    return
  fi
  if [[ -d /opt/rocm || -n "${ROCM_PATH:-}" ]]; then
    return
  fi
  log "ROCm was not detected from rocm-smi, /opt/rocm, or ROCM_PATH; continuing because the CPU benchmark can still run."
}

create_venv() {
  require_command python3
  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtual environment: ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
  else
    log "Using existing virtual environment: ${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip wheel setuptools
}

install_ffmpeg() {
  mkdir -p "${DOWNLOAD_DIR}" "${FFMPEG_DIR}"
  local archive="${DOWNLOAD_DIR}/ffmpeg-linux64.zip"

  download_file "${FFMPEG_ZIP_URL}" "${archive}"
  log "Extracting FFmpeg"
  rm -rf "${FFMPEG_DIR:?}/"*
  python - "${archive}" "${FFMPEG_DIR}" <<'PY'
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
target = Path(sys.argv[2])
with zipfile.ZipFile(archive) as zf:
    zf.extractall(target)
PY

  local ffmpeg_bin
  ffmpeg_bin="$(find "${FFMPEG_DIR}" -type f -path '*/bin/ffmpeg' -print -quit)"
  [[ -n "${ffmpeg_bin}" ]] || die "Could not find ffmpeg under ${FFMPEG_DIR}"
  chmod +x "${ffmpeg_bin}"
  chmod +x "$(dirname "${ffmpeg_bin}")"/ffprobe 2>/dev/null || true
  chmod +x "$(dirname "${ffmpeg_bin}")"/ffplay 2>/dev/null || true

  export PATH="$(dirname "${ffmpeg_bin}"):${PATH}"
  local activate_file="${VENV_DIR}/bin/activate"
  local marker="# Ambient AI FFmpeg path"
  if ! grep -Fq "${marker}" "${activate_file}"; then
    {
      printf '\n%s\n' "${marker}"
      printf 'export PATH="%s:$PATH"\n' "$(dirname "${ffmpeg_bin}")"
    } >> "${activate_file}"
  fi

  log "FFmpeg available at: $(command -v ffmpeg)"
  ffmpeg -version | head -n 1
}

python_tag() {
  python - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
}

install_ctranslate2_rocm() {
  mkdir -p "${DOWNLOAD_DIR}"
  local archive="${DOWNLOAD_DIR}/ctranslate2-rocm-${CTRANSLATE2_VERSION}-linux.zip"
  local wheels_dir="${DOWNLOAD_DIR}/ctranslate2-rocm-${CTRANSLATE2_VERSION}"
  local py_tag
  py_tag="$(python_tag)"

  download_file "${CTRANSLATE2_ROCM_ZIP_URL}" "${archive}"
  rm -rf "${wheels_dir}"
  mkdir -p "${wheels_dir}"
  log "Extracting CTranslate2 ROCm wheels"
  python - "${archive}" "${wheels_dir}" <<'PY'
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
target = Path(sys.argv[2])
with zipfile.ZipFile(archive) as zf:
    zf.extractall(target)
PY

  local wheel
  wheel="$(find "${wheels_dir}" -type f -name "ctranslate2-${CTRANSLATE2_VERSION}-${py_tag}-*linux*.whl" | sort | head -n 1)"
  if [[ -z "${wheel}" ]]; then
    find "${wheels_dir}" -type f -name '*.whl' -print
    die "No CTranslate2 ROCm wheel found for Python tag ${py_tag}. Try a Python version supported by the release."
  fi

  log "Installing CTranslate2 ROCm wheel: $(basename "${wheel}")"
  python -m pip install --force-reinstall "${wheel}"
}

install_faster_whisper() {
  log "Installing Faster Whisper runtime dependencies"
  python -m pip install -r "${ROOT_DIR}/requirements-faster-whisper-common.txt"

  log "Installing faster-whisper without dependencies so the ROCm CTranslate2 wheel is preserved"
  python -m pip install "faster-whisper==${FASTER_WHISPER_VERSION}" --no-deps
}

verify_python_backend() {
  log "Verifying installed Python packages"
  python - <<'PY'
import ctranslate2
import faster_whisper

print("ctranslate2:", ctranslate2.__version__)
print("faster_whisper:", getattr(faster_whisper, "__version__", "unknown"))
try:
    print("supported compute types on cuda:", ctranslate2.get_supported_compute_types("cuda"))
except Exception as exc:
    print("cuda/ROCm backend check failed:", exc)
PY
}

run_benchmark() {
  [[ -f "${AUDIO_FILE}" ]] || die "Audio file not found: ${AUDIO_FILE}. Save your test audio as scripts/audio.mp3 or set AUDIO_FILE=/path/to/file."

  local args=("${AUDIO_FILE}" "--model" "${MODEL_NAME}" "--beam-size" "${BEAM_SIZE}")
  if [[ -n "${LANGUAGE}" ]]; then
    args+=("--language" "${LANGUAGE}")
  fi

  log "Running CPU vs ROCm Faster Whisper benchmark"
  python "${SCRIPT_DIR}/compare_faster_whisper_cpu_rocm.py" "${args[@]}"
}

main() {
  ensure_linux
  ensure_rocm_visible
  create_venv
  install_ffmpeg
  install_ctranslate2_rocm
  install_faster_whisper
  verify_python_backend
  run_benchmark

  log "Done. To reuse this environment later:"
  printf 'source "%s/bin/activate"\n' "${VENV_DIR}"
}

main "$@"
