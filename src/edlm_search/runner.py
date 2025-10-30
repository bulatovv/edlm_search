import asyncio
import importlib.util
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Manager, Process
from pathlib import Path
from typing import Protocol

import pandas as pd
import torch
from zeus.monitor import ZeusMonitor

from edlm_search.candidate import Candidate


class Runner(Protocol):
    """A protocol defining the interface for running and managing candidate executions."""

    async def run(
        self,
        candidate: Candidate,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        run_args: dict | None = None,
    ):
        """
        Runs the candidate's code, yielding timestamped output lines.
        """
        ...

    def stop(self):
        """
        Stops the currently running candidate process.
        """
        ...


class QueueingStdoutInterceptor:
    """A stdout interceptor that puts captured lines into a queue."""

    def __init__(self, encoding, queue):
        self.encoding = encoding
        self.queue = queue
        self.buffer = ''

    def write(self, data):
        if isinstance(data, bytes):
            data = data.decode(self.encoding, 'replace')

        self.buffer += data
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self.queue.put({'timestamp': datetime.now(), 'stdout_line': line})

    def flush(self):
        pass

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.buffer:
            self.queue.put({'timestamp': datetime.now(), 'stdout_line': self.buffer})
        sys.stdout = self.original_stdout


def _run_candidate_in_process(
    candidate_files: dict[str, str],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    run_args: dict | None,
    queue,
    sentinel,
):
    """
    This function runs in a separate process and executes the candidate's code directly.
    """
    # Yield the initial timestamp from inside the process for better precision
    queue.put({'process_start_time': datetime.now()})
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write all candidate files to temp dir
            for filename, content in candidate_files.items():
                (temp_path / filename).write_text(content)

            # Save dataframes to parquet files
            train_data_path = temp_path / 'train.parquet'
            validation_data_path = temp_path / 'validation.parquet'
            train_df.to_parquet(train_data_path)
            validation_df.to_parquet(validation_data_path)

            # Prepare arguments for main function
            if run_args is None:
                run_args = {}
            run_args['train_data_path'] = str(train_data_path)
            run_args['valid_data_path'] = str(validation_data_path)

            original_cwd = os.getcwd()
            sys.path.insert(0, str(temp_path))
            os.chdir(temp_path)

            try:
                encoding = getattr(sys.stdout, 'encoding', 'utf-8')
                monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
                with QueueingStdoutInterceptor(encoding, queue):
                    # Dynamically import and run the candidate's main function
                    spec = importlib.util.spec_from_file_location('main', 'main.py')
                    if spec is None or spec.loader is None:
                        raise ImportError('Could not create module spec from main.py')

                    main_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(main_module)

                    if not hasattr(main_module, 'main'):
                        raise RuntimeError('Candidate code does not have a main() function.')
                    
                    monitor.begin_window('run')
                    main_module.main(**run_args)
                    mes = monitor.end_window('run')
                    queue.put({'total_energy_joules': mes.total_energy})

            finally:
                # Restore original state
                os.chdir(original_cwd)
                sys.path.remove(str(temp_path))
    finally:
        # Signal that execution is complete
        queue.put(sentinel)


class UnsafeRunner:
    """
    An unsafe runner that orchestrates model training in a separate, terminable process.
    """

    def __init__(self):
        self._process: Process | None = None

    async def run(
        self,
        candidate: Candidate,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        run_args: dict | None = None,
    ):
        """
        Runs the candidate's main() function in a separate process, yielding stdout lines as they are produced.
        """
        if self._process and self._process.is_alive():
            raise RuntimeError(
                'Another candidate is already running in this runner instance.'
            )

        loop = asyncio.get_running_loop()
        manager = Manager()
        queue = manager.Queue()
        sentinel = object()

        self._process = Process(
            target=_run_candidate_in_process,
            args=(
                candidate.files,
                train_df,
                validation_df,
                run_args,
                queue,
                sentinel,
            ),
        )
        self._process.start()

        try:
            with ThreadPoolExecutor() as pool:
                while True:
                    # Run the blocking queue.get() in a separate thread
                    item = await loop.run_in_executor(pool, queue.get)
                    if item is sentinel:
                        break
                    yield item
        finally:
            if self._process and self._process.is_alive():
                self._process.join(timeout=1)
            if self._process and self._process.is_alive():
                self._process.terminate()
            self._process = None

    def stop(self):
        """
        Stops the currently running candidate process.
        """
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process = None


class RunnerOutputParseError(Exception):
    pass
