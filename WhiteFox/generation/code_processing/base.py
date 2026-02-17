from abc import ABC, abstractmethod
from typing import List, Tuple


class CodeParser(ABC):

    @abstractmethod
    def split_func_tensor(self, code: str) -> Tuple[str, str, List[str], str]:
        ...

    @abstractmethod
    def preprocessing(self, code: str) -> str:
        ...

    @abstractmethod
    def process_code(self, code: str) -> str:
        ...
