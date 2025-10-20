from pathlib import Path


class Problem:
    def __init__(self, description: str):
        self.description: str = description
        ...

    @classmethod
    def from_directory(cls, dir: str):
        path = Path(dir)
        description = (path / 'problem.md').read_text()
        return cls(description=description)
