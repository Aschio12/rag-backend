import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        ...

    def log(self, message: str):
        logger.info(f"[{self.name}] {message}")
