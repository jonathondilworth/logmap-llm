import re
import sys

# default:
_RESET = "\033[0m"

# serious errors and potentially breaking warnings:
_BRIGHT_RED = "\033[91m"

# general-case warnings:
_PASTEL_YELLOW = "\033[38;5;229m"

# informative output:
_PASTEL_BLUE = "\033[38;5;117m"

# success output:
_GREEN = "\033[38;5;114m"

# strip ANSI escape characters for logging purposes
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def error(msg: str) -> None:
    print(f"{_BRIGHT_RED}[ERROR] {msg}{_RESET}")


def critical(msg: str) -> None:
    print(f"{_BRIGHT_RED}[CRITICAL WARNING] {msg}{_RESET}")


def warning(msg: str) -> None:
    print(f"{_PASTEL_YELLOW}[WARNING] {msg}{_RESET}")


def info(msg: str) -> None:
    print(f"{_PASTEL_BLUE}[INFO] {msg}{_RESET}")


def step(msg: str, enumeration:int|None = None) -> None:
    step_enumeratin = (f" {str(enumeration)}") if enumeration is not None else ""
    print(f"{_PASTEL_BLUE}[STEP{step_enumeratin}] {msg}{_RESET}")


def success(msg: str) -> None:
    print(f"{_GREEN}[COMPLETE] {msg}{_RESET}")


class TeeWriter:
    """
    Write to std::out and a log file.

    Terminal output preserves ANSI colour codes.
    The log-file output has all ANSI codes stripped.

    Implements sys.stdout interfaces:
        - isatty - encoding - fileno - writable
    """

    def __init__(self, log_path: str, original_stdout):
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.original_stdout = original_stdout

    def write(self, data: str) -> int:
        self.original_stdout.write(data)
        self.log_file.write(strip_ansi(data))
        return len(data)

    def flush(self) -> None:
        self.original_stdout.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()

    def isatty(self) -> bool:
        """delegates to the actual terminal"""
        return self.original_stdout.isatty()

    @property
    def encoding(self) -> str:
        """
        return the encoding in use by the terminal stream
        allows getEncoding, setEncoding as per @property
        """
        return getattr(self.original_stdout, "encoding", None) or "utf-8"

    def fileno(self) -> int:
        """return the file descriptor in use by the terminal stream"""
        return self.original_stdout.fileno()

    def writable(self) -> bool:
        return True
