import asyncio
import base64
import contextlib
import importlib.util
import json
import os
import pickle
import sys
import tempfile
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd
import torch
from tblib import pickling_support
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


class UnsafeRunner:
    """
    Runs a candidate in a separate, isolated Python process using asyncio.subprocess.

    This approach avoids the deadlocks associated with fork(), threading,
    and multiprocessing.Manager by starting a clean process and communicating
    over a dedicated pipe for structured data, leaving stdout for logs.
    """

    def __init__(self):
        self._process: asyncio.subprocess.Process | None = None
        self._comm_transport: asyncio.transports.ReadTransport | None = None
        self._stopped_by_user = False

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
        pickling_support.install()

        self._stopped_by_user = False

        if self._process:
            raise RuntimeError('Another candidate is already running in this runner instance.')

        if run_args is None:
            run_args = {}

        comm_r, comm_w = os.pipe()

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

                # 4. Set environment variable and start the subprocess
                child_env = os.environ.copy()
                child_env['ZEUS_LOG_LEVEL'] = 'CRITICAL'
                child_env['COMM_PIPE_FD'] = str(comm_w)

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
                    env=child_env,
                    pass_fds=[comm_w],
                )
                os.close(comm_w)  # Close write end in parent

                # 5. Asynchronously read from both communication pipe and stdout
                loop = asyncio.get_running_loop()
                comm_pipe_reader = asyncio.StreamReader()
                self._comm_transport, _ = await loop.connect_read_pipe(
                    lambda: asyncio.StreamReaderProtocol(comm_pipe_reader), os.fdopen(comm_r)
                )

                tasks = {
                    asyncio.create_task(self._process.stdout.readline()): 'stdout',
                    asyncio.create_task(comm_pipe_reader.readline()): 'comm',
                }

                while tasks:
                    done, pending = await asyncio.wait(
                        tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                    )

                    for future in done:
                        source = tasks.pop(future)
                        line_bytes = future.result()

                        if not line_bytes:  # EOF
                            continue

                        if source == 'stdout':
                            line = line_bytes.decode('utf-8', 'replace').rstrip()
                            yield {'timestamp': datetime.now(), 'stdout_line': line}
                            tasks[asyncio.create_task(self._process.stdout.readline())] = (
                                'stdout'
                            )

                        elif source == 'comm':
                            line = line_bytes.decode('utf-8', 'replace').rstrip()
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError as e:
                                raise RunnerOutputParseError(
                                    f'Could not parse JSON from child: {line}'
                                ) from e

                            event_type = event.get('type')

                            if event_type == 'exception':
                                try:
                                    exc_info_bytes = base64.b64decode(
                                        event['data']['exc_info']
                                    )
                                    exc_info = pickle.loads(exc_info_bytes)
                                except Exception as e:
                                    raise RunnerOutputParseError(
                                        f'Failed to unpickle exception from child: {e}'
                                    ) from e
                                raise exc_info[1].with_traceback(exc_info[2])

                            try:
                                if event_type == 'start':
                                    ts_str = event['data']['process_start_time']
                                    yield {
                                        'process_start_time': datetime.fromisoformat(ts_str)
                                    }
                                elif event_type == 'zeus':
                                    yield event['data']
                                elif event_type == 'epoch_result':
                                    yield {
                                        'timestamp': datetime.now(),
                                        'epoch_result': event['data']['predictions'],
                                    }
                            except KeyError as e:
                                raise RunnerOutputParseError(
                                    f'Invalid event data from child: {line}'
                                ) from e
                            tasks[asyncio.create_task(comm_pipe_reader.readline())] = 'comm'

                # 6. Wait for the process to exit and check for errors
                return_code = await self._process.wait()
                if return_code != 0 and not self._stopped_by_user:
                    raise RunnerOutputParseError(
                        f'Candidate process exited with non-zero code: {return_code}'
                    )

        finally:
            # 7. Cleanup
            if self._comm_transport:
                self._comm_transport.close()
            else:
                os.close(comm_r)
            self._comm_transport = None
            if self._process and self._process.returncode is None:
                try:
                    self._process.terminate()
                    await self._process.wait()
                except ProcessLookupError:
                    pass  # Process already finished
            self._process = None

    def stop(self):
        """Stops the currently running candidate process."""
        if self._process and self._process.returncode is None:
            self._stopped_by_user = True
            with contextlib.suppress(ProcessLookupError):
                self._process.terminate()


def _execute_candidate_script(temp_dir: str, run_args: dict):
    """
    Execute the candidate script in a subprocess.

    This synchronous function runs *inside the subprocess*. It imports and
    executes the candidate's code, which is expected to be a generator. It
    sends structured events (start, results, metrics, exceptions) back to the
    parent process over a dedicated communication pipe.
    """
    pickling_support.install()
    comm_fd = int(os.environ['COMM_PIPE_FD'])
    # The parent process is responsible for closing the read end of the pipe.
    # The child process only writes to it.
    with os.fdopen(comm_fd, 'w') as comm_pipe:

        class CustomEncoder(json.JSONEncoder):
            def default(self, o):
                if hasattr(o, 'tolist'):
                    return o.tolist()
                if hasattr(o, 'item'):  # for numpy scalars
                    return o.item()
                return super().default(o)

        def send_event(event_type: str, data: dict):
            """Sends a JSON-encoded event to the parent process."""
            try:
                event = {'type': event_type, 'data': data}
                json.dump(event, comm_pipe, cls=CustomEncoder)
                comm_pipe.write('\n')
                comm_pipe.flush()
            except (OSError, BrokenPipeError):
                # Parent process might have terminated, so we can't send events.
                # Exit gracefully.
                sys.exit(1)

        original_cwd = os.getcwd()
        temp_path = Path(temp_dir)
        monitor = None

        try:
            # 1. Set up the environment
            os.chdir(temp_path)
            sys.path.insert(0, str(temp_path))

            # 2. Signal start time to parent
            send_event('start', {'process_start_time': datetime.now().isoformat()})

            # 3. Set up monitoring
            monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

            # 4. Dynamically import and run the candidate's main function
            spec = importlib.util.spec_from_file_location('main', 'main.py')
            if spec is None or spec.loader is None:
                raise ImportError('Could not create module spec from main.py')

            main_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main_module)

            if not hasattr(main_module, 'main'):
                raise RuntimeError('Candidate code does not have a main() function.')

            main_generator = main_module.main(**run_args)

            monitor.begin_window('run')
            for epoch_predictions in main_generator:
                send_event('epoch_result', {'predictions': epoch_predictions})

        except Exception:
            # Report exception to parent and exit with error
            exc_info = sys.exc_info()
            exc_info_bytes = pickle.dumps(exc_info)
            exc_info_b64 = base64.b64encode(exc_info_bytes).decode('utf-8')
            send_event('exception', {'exc_info': exc_info_b64})
            sys.exit(1)
        else:
            # 5. Report measurements and exit cleanly on success
            if monitor:
                mes = monitor.end_window('run')
                result = {'total_energy_joules': mes.total_energy}
                send_event('zeus', result)
            sys.exit(0)
        finally:
            # 6. Restore original state
            os.chdir(original_cwd)
            if str(temp_path) in sys.path:
                sys.path.remove(str(temp_path))


if __name__ == '__main__':
    """
    This block is executed when the module is run as a script
    (e.g., `python -m my_module_name ...`)
    """
    pickling_support.install()
    parser = ArgumentParser()
    parser.add_argument(
        '--temp-dir', required=True, help='Temporary directory with candidate files'
    )
    parser.add_argument('--run-args', required=True, help='JSON string of run arguments')
    args = parser.parse_args()

    try:
        run_args_dict = json.loads(args.run_args)
    except Exception:
        # Try to report the specific error over the comm pipe.
        # If this fails, the parent will catch the non-zero exit code.
        comm_fd_str = os.environ.get('COMM_PIPE_FD')
        if comm_fd_str:
            with contextlib.suppress(OSError, BrokenPipeError, ValueError):
                comm_fd = int(comm_fd_str)
                with os.fdopen(comm_fd, 'w') as comm_pipe:
                    exc_info = sys.exc_info()
                    exc_info_bytes = pickle.dumps(exc_info)
                    exc_info_b64 = base64.b64encode(exc_info_bytes).decode('utf-8')
                    event = {
                        'type': 'exception',
                        'data': {'exc_info': exc_info_b64},
                    }
                    json.dump(event, comm_pipe)
                    comm_pipe.write('\n')
                    comm_pipe.flush()
        sys.exit(1)

    _execute_candidate_script(args.temp_dir, run_args_dict)
