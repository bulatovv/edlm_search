from pathlib import Path


class Problem:
    """Represents a problem statement."""

    def __init__(self, statement: str):
        self.statement: str = statement
        ...

    @classmethod
    def from_directory(cls, dir: str):
        """Load a problem statement from a file in a directory."""
        path = Path(dir)
        statement = (path / 'statement.md').read_text()
        return cls(statement=statement)
