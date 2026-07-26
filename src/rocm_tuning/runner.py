from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class LlamaServerProcess:
    process: subprocess.Popen
    base_url: str

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


class LlamaServerRunner:
    def __init__(self, *, llama_server_path: str, host: str = "127.0.0.1", port: int = 8091, api_key: str = "testkey"):
        self.llama_server_path = str(llama_server_path)
        self.host = host
        self.port = int(port)
        self.api_key = api_key

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, *, models_preset: str | Path, model: str, extra_args: list[str] | None = None) -> LlamaServerProcess:
        executable = Path(self.llama_server_path)
        if not executable.is_file():
            raise FileNotFoundError(f"llama-server not found: {executable}")
        preset = Path(models_preset)
        if not preset.is_file():
            raise FileNotFoundError(f"models preset not found: {preset}")
        args = [
            str(executable),
            "--host", self.host,
            "--port", str(self.port),
            "--api-key", self.api_key,
            "--models-preset", str(preset),
            "--model", model,
        ]
        if extra_args:
            args.extend(extra_args)
        process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        server = LlamaServerProcess(process=process, base_url=self.base_url)
        self._wait_until_ready(server)
        return server

    def _wait_until_ready(self, server: LlamaServerProcess, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            if server.process.poll() is not None:
                stderr = ""
                if server.process.stderr is not None:
                    try:
                        stderr = server.process.stderr.read()[-2000:]
                    except Exception:
                        stderr = ""
                raise RuntimeError(f"llama-server exited before becoming ready: {stderr}")
            try:
                response = requests.get(f"{server.base_url}/v1/models", timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException as exc:
                last_error = exc
            time.sleep(0.5)
        server.stop()
        raise RuntimeError(f"llama-server did not become ready on {server.base_url}: {last_error}")
