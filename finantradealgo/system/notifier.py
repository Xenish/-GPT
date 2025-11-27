from typing import Protocol


class Notifier(Protocol):
    def info(self, msg: str) -> None:
        ...

    def warn(self, msg: str) -> None:
        ...

    def critical(self, msg: str) -> None:
        ...
