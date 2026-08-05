import configparser
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_USER_DATA_DIR = Path(
    os.environ.get("AMBIENT_USER_DATA_DIR", str(Path.home() / "AmbientAI" / "data"))
).expanduser()
_configured_path = os.environ.get("AMBIENT_CONFIG_PATH", "").strip()
CONFIG_PATH = Path(_configured_path).expanduser() if _configured_path else PROJECT_ROOT / "config.ini"
if not CONFIG_PATH.exists():
    CONFIG_PATH = PROJECT_ROOT / "config.example.ini"


class AppConfig:
    def __init__(self, path: Path = CONFIG_PATH):
        self.path = path
        self.parser = configparser.ConfigParser()
        self.parser.read(path, encoding="utf-8")

    def get_str(self, section: str, option: str, fallback: str) -> str:
        return self.parser.get(section, option, fallback=fallback)

    def get_int(self, section: str, option: str, fallback: int) -> int:
        return self.parser.getint(section, option, fallback=fallback)

    def get_float(self, section: str, option: str, fallback: float) -> float:
        return self.parser.getfloat(section, option, fallback=fallback)

    def get_bool(self, section: str, option: str, fallback: bool) -> bool:
        return self.parser.getboolean(section, option, fallback=fallback)

    def get_model(self, option: str, fallback: str, section: str = "models") -> str:
        value = self.parser.get(section, option, fallback=fallback)
        normalized = str(value).strip()
        return normalized or fallback


CONFIG = AppConfig()
