import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error

from edlm_search.candidate import Candidate
from edlm_search.evaluator import Evaluator
from edlm_search.problem import Problem
from edlm_search.runner import UnsafeRunner

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
        scores = []
        async for predictions in runner.run(
            candidate=candidate,
            train_df=self._train_df,
            validation_df=self._validation_df,
            num_epochs=5,  # Assuming 5 epochs for evaluation
        ):
            score = mean_squared_error(self._ground_truth, predictions)
            scores.append(score)

        # For simplicity, returning the last score as the primary metric
        # In a real scenario, you might average scores, take the best, etc.
        return {'mean_squared_error': scores[-1]}
