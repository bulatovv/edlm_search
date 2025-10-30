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
        full_df = pd.read_csv(os.path.join(problem_dir, 'ETTm1.csv')).head(10000)
        
        train_size = int(len(full_df) * 0.8)
        self._train_df = full_df.iloc[:train_size]
        self._validation_df = full_df.iloc[train_size:]

    async def evaluate(self, runner: UnsafeRunner, candidate: Candidate) -> dict[str, float]:
        """
        Tests the candidate's ability to forecast pred_len steps given seq_len of context.
        """
        # Define arbitrary sequence and prediction lengths as per user instruction
        seq_len = 96
        pred_len = 24

        run_args = {'num_epochs': 20}
        
        # Construct the validation dataframe for the candidate from the validation set
        # It contains seq_len of context from the start of the validation set...
        context_df = self._validation_df.head(seq_len).copy()
        
        # ...followed by a scaffold for the prediction period.
        prediction_scaffold_df = self._validation_df.iloc[seq_len:seq_len + pred_len].copy()
        
        # The ground truth corresponds to the prediction period
        ground_truth = prediction_scaffold_df['OT'].values
        
        # Mask all feature and target columns in the prediction period
        features_to_mask = [col for col in prediction_scaffold_df.columns if col != 'date']
        prediction_scaffold_df[features_to_mask] = np.nan
        
        validation_df_for_candidate = pd.concat([context_df, prediction_scaffold_df])

        launch_time, output_lines = await runner.run(
            candidate=candidate,
            train_df=self._train_df,
            validation_df=validation_df_for_candidate,
            run_args=run_args,
        )

        if not output_lines:
            raise RunnerOutputParseError("No output received from the runner.")

        # The last line should contain the single forecast of pred_len steps
        _timestamp, last_line = output_lines[-1]
        
        try:
            predictions = ast.literal_eval(last_line)
            predictions = np.array(predictions)

            # Truncate predictions and ground_truth to the minimum length
            min_len = min(len(predictions), len(ground_truth))
            if min_len == 0:
                raise ValueError("Prediction or ground truth is empty.")
            
            predictions = predictions[:min_len]
            ground_truth = ground_truth[:min_len]

            score = mean_squared_error(ground_truth, predictions)
        except (ValueError, SyntaxError) as e:
            raise RunnerOutputParseError(
                f"Failed to parse runner output line: '{last_line}'"
            ) from e

        return {'mean_squared_error': score}