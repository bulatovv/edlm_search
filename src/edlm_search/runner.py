import logging
import os
import sys
import tempfile
from argparse import ArgumentParser
from datetime import datetime
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
        """Runs the candidate's code, yielding timestamped output lines."""
        ...

    def stop(self):
        """Stops the currently running candidate process."""
        ...


class RunnerOutputParseError(Exception):
    """Raised when the runner's subprocess fails or produces invalid output."""

    pass


class AsyncSubprocessRunner:
    """
    Runs a candidate in a separate, isolated Python process using asyncio.subprocess.

    This approach avoids the deadlocks associated with fork(), threading,
    and multiprocessing.Manager by starting a clean process and communicating
    over standard pipes.
    """

    def __init__(self):
        self._process: asyncio.subprocess.Process | None = None

    async def run(
        self,
        candidate: Candidate,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        run_args: dict | None = None,
    ):
        """
        Runs the candidate's main() function in a separate process.

        Yields metrics and stdout lines as they are produced.
        """
        if self._process:
            raise RuntimeError('Another candidate is already running in this runner instance.')

        if run_args is None:
            run_args = {}

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # 1. Write all candidate files to temp dir
                for filename, content in candidate.files.items():
                    (temp_path / filename).write_text(content)

                # 2. Save dataframes to parquet files
                train_data_path = temp_path / 'train.parquet'
                validation_data_path = temp_path / 'validation.parquet'
                train_df.to_parquet(train_data_path)
                validation_df.to_parquet(validation_data_path)

                # 3. Prepare arguments for the subprocess
                run_args['train_data_path'] = str(train_data_path)
                run_args['valid_data_path'] = str(validation_data_path)
                run_args_json = json.dumps(run_args)

                # 4. Start the subprocess
                # We run this very module as a script (__name__)
                self._process = await asyncio.create_subprocess_exec(
                    sys.executable,  # The current python interpreter
                    '-m',
                    __name__,  # Run this file as a module
                    '--temp-dir',
                    str(temp_path),
                    '--run-args',
                    run_args_json,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                )

                # 5. Asynchronously read output from the process
                start_time_found = False
                while True:
                    line_bytes = await self._process.stdout.readline()
                    if not line_bytes:
                        break  # End of stream

                    line = line_bytes.decode('utf-8', 'replace').rstrip()

                    if not start_time_found:
                        if line.startswith('RUNNER_EVENT:PROCESS_START_TIME:'):
                            start_time_found = True
                            ts_str = line.split(':', 2)[-1]
                            yield {'process_start_time': datetime.fromisoformat(ts_str)}
                        # Discard any other lines until start time is found
                        continue

                    # After start time is found, process normally
                    # (but skip reprocessing the start time event itself)
                    if line.startswith('RUNNER_EVENT:PROCESS_START_TIME:'):
                        continue

                    elif line.startswith('RUNNER_EVENT:ZEUS_MEASUREMENT:'):
                        json_data = line.split(':', 2)[-1]
                        yield json.loads(json_data)

                    elif line.startswith('RUNNER_EVENT:EXCEPTION:'):
                        error_msg = line.split(':', 2)[-1]
                        raise RunnerOutputParseError(f'Candidate script failed: {error_msg}')

                    else:
                        # Yield regular stdout lines
                        yield {'timestamp': datetime.now(), 'stdout_line': line}

                # 6. Wait for the process to exit and check for errors
                return_code = await self._process.wait()
                if return_code != 0:
                    raise RunnerOutputParseError(
                        f'Candidate process exited with non-zero code: {return_code}'
                    )

        finally:
            # 7. Cleanup
            if self._process and self._process.returncode is None:
                # Process is still running, terminate it
                try:
                    self._process.terminate()
                    await self._process.wait()
                except ProcessLookupError:
                    pass  # Process already finished
            self._process = None

    def stop(self):
        """Stops the currently running candidate process."""
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
            except ProcessLookupError:
                pass  # Process already dead
            self._process = None


def _execute_candidate_script(temp_dir: str, run_args: dict):
    """
    This is the synchronous function that runs *inside the subprocess*.

    It imports and executes the candidate's code and prints measurements
    to stdout for the parent process to capture.
    """
    # We are now in a clean, separate process.
    # The parent process set our CWD.
    original_cwd = os.getcwd()
    temp_path = Path(temp_dir)

    try:
        # 1. Set up the environment
        os.chdir(temp_path)
        sys.path.insert(0, str(temp_path))

        # 2. Signal start time to parent
        # flush=True is critical so the parent receives messages immediately
        print(f'RUNNER_EVENT:PROCESS_START_TIME:{datetime.now().isoformat()}', flush=True)

        # 3. Set up monitoring
        logging.getLogger('zeus').setLevel(logging.CRITICAL)
        monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

        # 4. Dynamically import and run the candidate's main function
        spec = importlib.util.spec_from_file_location('main', 'main.py')
        if spec is None or spec.loader is None:
            raise ImportError('Could not create module spec from main.py')

        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)

        if not hasattr(main_module, 'main'):
            raise RuntimeError('Candidate code does not have a main() function.')

        monitor.begin_window('run')
        main_module.main(**run_args)

    except Exception as e:
        # Report exception to parent and exit with error
        print(f'RUNNER_EVENT:EXCEPTION:{e}', flush=True)
        sys.exit(1)

    finally:
        # 5. Report measurements and exit cleanly
        mes = monitor.end_window('run')
        result = {'total_energy_joules': mes.total_energy}
        print(f'RUNNER_EVENT:ZEUS_MEASUREMENT:{json.dumps(result)}', flush=True)

        # Restore original state
        os.chdir(original_cwd)
        if str(temp_path) in sys.path:
            sys.path.remove(str(temp_path))

        sys.exit(0)


if __name__ == '__main__':
    """
    This block is executed when the module is run as a script
    (e.g., `python -m my_module_name ...`)
    """
    parser = ArgumentParser()
    parser.add_argument(
        '--temp-dir', required=True, help='Temporary directory with candidate files'
    )
    parser.add_argument('--run-args', required=True, help='JSON string of run arguments')
    args = parser.parse_args()

    try:
        run_args_dict = json.loads(args.run_args)
    except json.JSONDecodeError as e:
        print(f'RUNNER_EVENT:EXCEPTION:Failed to decode run-args JSON: {e}', flush=True)
        sys.exit(1)

    _execute_candidate_script(args.temp_dir, run_args_dict)
