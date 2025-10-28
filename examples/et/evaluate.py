import ast
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error

from edlm_search.candidate import Candidate
from edlm_search.evaluator import Evaluator
from edlm_search.problem import Problem
from edlm_search.runner import RunnerOutputParseError, UnsafeRunner

load_dotenv()


class ETEvaluator(Evaluator):
    def __init__(self, problem_dir: str):
        self._problem_dir = problem_dir
        self._problem = Problem.from_directory(problem_dir)
        full_train_df = pd.read_csv(os.path.join(problem_dir, 'ETTm1.csv'))
        self._train_df = full_train_df.iloc[:-100]
        self._validation_df = full_train_df.iloc[-100:]
        self._ground_truth = self._validation_df['OT'].values

    async def evaluate(self, runner: UnsafeRunner, candidate: Candidate) -> dict[str, float]:
        """Runs the candidate using the provided runner and calculates problem-specific metrics."""
        run_args = {'num_epochs': 20}
        launch_time, output_lines = await runner.run(
            candidate=candidate,
            train_df=self._train_df,
            validation_df=self._validation_df,
            run_args=run_args,
        )

        scores = []
        for _timestamp, line in output_lines:
            try:
                # The line is a string representation of a list of predictions
                predictions = ast.literal_eval(line)
                predictions = np.array(predictions)
                score = mean_squared_error(self._ground_truth, predictions)
                scores.append(score)
            except (ValueError, SyntaxError) as e:
                raise RunnerOutputParseError(
                    f"Failed to parse runner output line: '{line}'"
                ) from e

        if not scores:
            return {'mean_squared_error': float('inf')}

        # For simplicity, returning the last score as the primary metric
        # In a real scenario, you might average scores, take the best, etc.
        return {'mean_squared_error': scores[-1]}
