import importlib.util
import inspect
import json
import sys
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from edlm_search.candidate import Candidate


class UnsafeRunner:
    """A simple, non-secure runner that orchestrates model training in the same process."""

    def __init__(self):
        pass

    async def run(
        self,
        candidate: Candidate,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        num_epochs: int,
    ) -> AsyncGenerator[np.ndarray]:
        """A generator that runs the training process and yields predictions for each epoch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sys.path.insert(0, temp_dir)
            try:
                # Write model file to temp dir to be able to import it.
                (temp_path / 'model.py').write_text(candidate.files['model.py'])

                # Dynamically import the model from the temporary directory
                spec = importlib.util.spec_from_file_location('model', temp_path / 'model.py')
                assert spec and spec.loader
                model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_module)

                # Find Dataset and Model classes
                dataset_class = None
                model_class = None
                for _name, obj in inspect.getmembers(model_module):
                    if inspect.isclass(obj):
                        if (
                            issubclass(obj, torch.utils.data.Dataset)
                            and obj is not torch.utils.data.Dataset
                        ):
                            dataset_class = obj
                        elif issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
                            model_class = obj

                if not dataset_class or not model_class:
                    raise RuntimeError('Could not find Dataset or Model class in model.py')

                # Load configurations and data
                model_config = json.loads(candidate.files['model_config.json'])
                training_args = json.loads(candidate.files['training_args.json'])

                # Instantiate dataset, model, and trainer
                dataset = dataset_class.from_config(df=train_df, config=model_config)
                model = model_class.from_config(model_config)
                trainer = model_module.Trainer.from_config(
                    num_epochs=num_epochs, config=training_args, model=model, dataset=dataset
                )

                # Training and prediction loop
                train_generator = trainer.train()
                for _ in range(num_epochs):
                    next(train_generator)  # Train for one epoch
                    predictions = trainer.predict(validation_df)
                    yield predictions
            finally:
                # Clean up sys.path
                sys.path.remove(temp_dir)
