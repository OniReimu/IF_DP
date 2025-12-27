import os
import sys
from typing import Optional


# ANSI color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# Logging levels
class LogLevel:
    ERROR = 0
    WARN = 1
    INFO = 2
    DEBUG = 3
    TRACE = 4


class ColoredLogger:
    def __init__(self, name: str, level: int = LogLevel.INFO) -> None:
        self.name = name
        self.level = level
        self.enabled = True

    def _maybe_format(self, message: str, args: tuple[object, ...]) -> str:
        if not args:
            return message
        try:
            return message % args
        except Exception:
            joined = " ".join(str(a) for a in args)
            return f"{message} {joined}".rstrip()

    def _format_message(self, level: str, color: str, message: str) -> str:
        if not self.enabled or not sys.stdout.isatty():
            return f"[{self.name}] {level}: {message}"
        return f"{color}[{self.name}] {level}: {message}{Colors.END}"

    def error(self, message: str, *args: object) -> None:
        if self.level >= LogLevel.ERROR:
            print(self._format_message("ERROR", Colors.RED + Colors.BOLD, self._maybe_format(message, args)))

    def warn(self, message: str, *args: object) -> None:
        if self.level >= LogLevel.WARN:
            print(self._format_message("WARN", Colors.YELLOW + Colors.BOLD, self._maybe_format(message, args)))

    def info(self, message: str, *args: object) -> None:
        if self.level >= LogLevel.INFO:
            print(self._format_message("INFO", Colors.BLUE, self._maybe_format(message, args)))

    def debug(self, message: str, *args: object) -> None:
        if self.level >= LogLevel.DEBUG:
            print(self._format_message("DEBUG", Colors.CYAN, self._maybe_format(message, args)))

    def trace(self, message: str, *args: object) -> None:
        if self.level >= LogLevel.TRACE:
            print(self._format_message("TRACE", Colors.MAGENTA, self._maybe_format(message, args)))

    def success(self, message: str, *args: object) -> None:
        if self.level >= LogLevel.INFO:
            print(self._format_message("SUCCESS", Colors.GREEN + Colors.BOLD, self._maybe_format(message, args)))

    def highlight(self, message: str, *args: object) -> None:
        if self.level >= LogLevel.INFO:
            print(
                self._format_message("HIGHLIGHT", Colors.WHITE + Colors.BOLD + Colors.UNDERLINE, self._maybe_format(message, args))
            )


class LoggingConfig:
    def __init__(self) -> None:
        self.debug_enabled = self._get_debug_from_env()
        self.level = LogLevel.TRACE if self.debug_enabled else LogLevel.INFO
        self.loggers: dict[str, ColoredLogger] = {}

    def _get_debug_from_env(self) -> bool:
        debug_vars = [
            "DEBUG",
            "LOG_DEBUG",
            "ANTI_NSO_DEBUG",
        ]
        for var in debug_vars:
            value = os.getenv(var, "").strip().lower()
            if value in {"1", "true", "yes", "y", "on"}:
                return True
        return False

    def get_logger(self, name: str, level: Optional[int] = None) -> ColoredLogger:
        if name in self.loggers:
            return self.loggers[name]
        use_level = self.level if level is None else int(level)
        logger = ColoredLogger(name, level=use_level)
        self.loggers[name] = logger
        return logger


_LOGGING_CONFIG = LoggingConfig()


def get_logger(name: str, level: Optional[int] = None) -> ColoredLogger:
    return _LOGGING_CONFIG.get_logger(name, level=level)
