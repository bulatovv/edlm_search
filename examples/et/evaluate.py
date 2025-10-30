import ast
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import auc, mean_squared_error

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
        context_df = self._validation_df.head(seq_len).copy()
        prediction_scaffold_df = self._validation_df.iloc[seq_len:seq_len + pred_len].copy()
        ground_truth = prediction_scaffold_df['OT'].values
        
        features_to_mask = [col for col in prediction_scaffold_df.columns if col != 'date']
        prediction_scaffold_df[features_to_mask] = np.nan
        
        validation_df_for_candidate = pd.concat([context_df, prediction_scaffold_df])

        launch_timestamp, output_lines = await runner.run(
            candidate=candidate,
            train_df=self._train_df,
            validation_df=validation_df_for_candidate,
            run_args=run_args,
        )

        if not output_lines:
            raise RunnerOutputParseError("No output received from the runner.")

        elapsed_times = []
        losses = []

        for timestamp, line in output_lines:
            try:
                predictions = np.array(ast.literal_eval(line))
                
                loss = mean_squared_error(ground_truth, predictions)
                elapsed_time = (timestamp - launch_timestamp).total_seconds()
                losses.append(loss)
                elapsed_times.append(elapsed_time)

            except (ValueError, SyntaxError):
                raise RunnerOutputParseError("cannot parse candidate output") 
        
        if len(elapsed_times) < 2:
            raise RunnerOutputParseError(
                "Could not parse at least two valid prediction outputs to calculate AUC."
            )

        # Calculate the Area Under the Curve for the loss vs. time plot
        loss_auc = auc(elapsed_times, losses)

        return {'loss_vs_time_auc': loss_auc}
