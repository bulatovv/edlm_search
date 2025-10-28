import importlib.util
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from edlm_search.candidate import Candidate


class StdoutInterceptor:
    def __init__(self, encoding):
        self.encoding = encoding
        self.buffer = ''
        self.timestamps_and_lines = []

    def write(self, data):
        if isinstance(data, bytes):
            data = data.decode(self.encoding, 'replace')

        self.buffer += data
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self.timestamps_and_lines.append((datetime.now(), line))

    def flush(self):
        pass


class UnsafeRunner:
    """A simple, non-secure runner that orchestrates model training in the same process."""

    def __init__(self):
        pass

    async def run(
        self,
        candidate: Candidate,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        run_args: dict | None = None,
    ) -> tuple[datetime, list[tuple[datetime, str]]]:
        """Runs the candidate's main() function and captures its stdout."""
        launch_timestamp = datetime.now()

        original_stdout = sys.stdout
        interceptor = StdoutInterceptor(getattr(original_stdout, 'encoding', 'utf-8'))
        sys.stdout = interceptor

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                original_cwd = os.getcwd()
                os.chdir(temp_path)
                sys.path.insert(0, str(temp_path))

                try:
                    # Write all candidate files to temp dir
                    for filename, content in candidate.files.items():
                        (temp_path / filename).write_text(content)

                    # Dynamically import the main script from the temporary directory
                    spec = importlib.util.spec_from_file_location('main', 'main.py')
                    if spec is None or spec.loader is None:
                        raise ImportError('Could not create module spec from main.py')

                    main_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(main_module)

                    if not hasattr(main_module, 'main'):
                        raise RuntimeError('Candidate code does not have a main() function.')

                    # The main function is expected to print predictions to stdout.
                    if run_args is None:
                        run_args = {}
                    main_module.main(train_df, validation_df, **run_args)

                finally:
                    os.chdir(original_cwd)
                    sys.path.remove(str(temp_path))
        finally:
            sys.stdout = original_stdout
            # Handle any remaining text in the buffer that doesn't end with a newline
            if interceptor.buffer:
                interceptor.timestamps_and_lines.append((datetime.now(), interceptor.buffer))

        return (launch_timestamp, interceptor.timestamps_and_lines)


class RunnerOutputParseError(Exception):
    pass
