import configparser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"


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


CONFIG = AppConfig()
