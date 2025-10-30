import ast
import json
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

        # Load training_args to get seq_len and pred_len
        with open(os.path.join(problem_dir, 'training_args.json'), 'r') as f:
            training_args = json.load(f)
        self._seq_len = training_args['sequence_length']
        self._pred_len = training_args['prediction_length']

        full_df = pd.read_csv(os.path.join(problem_dir, 'ETTm1.csv'))
        # The validation set is the 100 rows after the training set
        self._train_df = full_df.iloc[:-100]
        self._validation_df = full_df.iloc[-100:]
        
        # The ground truth is the first pred_len values of the validation set's target
        self._ground_truth = self._validation_df['OT'].iloc[:self._pred_len].values

    async def evaluate(self, runner: UnsafeRunner, candidate: Candidate) -> dict[str, float]:
        """
        Tests the candidate's ability to forecast pred_len steps given seq_len of context.
        """
        run_args = {'num_epochs': 20}
        
        # Construct the validation dataframe for the candidate
        # It contains seq_len of context from the end of the training set...
        context_df = self._train_df.tail(self._seq_len)
        
        # ...followed by a scaffold for the prediction period.
        prediction_scaffold_df = self._validation_df.head(self._pred_len).copy()
        
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

            if len(predictions) != self._pred_len:
                raise ValueError(f"Prediction length ({len(predictions)}) does not match expected pred_len ({self._pred_len})")

            score = mean_squared_error(self._ground_truth, predictions)
        except (ValueError, SyntaxError) as e:
            raise RunnerOutputParseError(
                f"Failed to parse runner output line: '{last_line}'"
            ) from e

        return {'mean_squared_error': score}