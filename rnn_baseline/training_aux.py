import torch
import numpy as np


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""

    def __init__(self, patience=7, mode='min', verbose=False, delta=0, save_path='checkpoint.hdf5', metric_name=None, save_format='torch'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            mode (str): One of ['min', 'max'], whether to maximize or minimaze the metric.
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_path (str): Path to saved model
        """
        if mode not in ['min', 'max']:
            raise ValueError(f'Unrecognized mode: {mode}!')

        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_prev_score = np.Inf if mode == 'min' else -np.Inf
        self.delta = delta
        self.save_path = save_path
        self.metric_name = 'metric' if not metric_name else metric_name
        if save_format not in ['torch', 'tf']:
            raise ValueError('Expected to save in one of the following formats: ["torch", "tf"]')
        self.save_format = save_format
        
    def __call__(self, metric_value, model):

        score = -metric_value if self.mode == 'min' else metric_value

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_value, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'No imporvement in Validation {self.metric_name}. Current: {score:.6f}. Current best: {self.best_score:.6f}')
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric_value, model)
            self.counter = 0

    def save_checkpoint(self, metric_value, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f'Validation {self.metric_name} improved ({self.best_prev_score:.6f} --> {metric_value:.6f}).  Saving model ...')
        if self.save_format == 'tf':
            model.save_weights(self.save_path)
        else:
            torch.save(model.state_dict(), self.save_path)
            
        self.best_prev_score = metric_value
