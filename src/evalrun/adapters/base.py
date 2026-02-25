from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    tool_meta: dict | None = None


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "", **params) -> GenerationResult:
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        ...
