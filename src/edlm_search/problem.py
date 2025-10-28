from pathlib import Path


class Problem:
    def __init__(self, statement: str):
        self.statement: str = statement
        ...

    @classmethod
    def from_directory(cls, dir: str):
        path = Path(dir)
        statement = (path / 'statement.md').read_text()
        return cls(statement=statement)
