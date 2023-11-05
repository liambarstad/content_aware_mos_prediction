import os
import torch
from typing import List
from datetime import datetime
import numpy as np
from scipy.stats import pearsonr, spearmanr

RUNS_DIR = os.environ['RUNS_DIR']

class Metric:
    '''
        base class for metrics, creates a list of predictions and targets 
        "calculate" function can then be overwritten to calculate the metric based on the predictions and targets
    '''
    def __init__(self):
        self.predictions = []
        self.targets = []

    def calculate(self):
        # overwrite
        pass

    def add_value(self, prediction: float, target: float):
        self.predictions.append(prediction)
        self.targets.append(target)

    def clear(self):
        self.predictions = []
        self.targets = []


class LCC(Metric):
    '''
        linear correlation coefficient
    '''
    def calculate(self):
        return pearsonr(self.predictions, self.targets)[0]


class SRCC(Metric):
    '''
        spearman rank correlation coefficient
    '''
    def calculate(self):
        return spearmanr(self.predictions, self.targets)[0]


class MSE(Metric):
    '''
        mean squared error
    '''
    def calculate(self):
        return ((np.array(self.predictions) - np.array(self.targets)) ** 2).mean()


class MOSNetMetrics:
    def __init__(self, name: str):
        self.name = name
        self.utterance_lcc, self.frame_lcc = [LCC(), LCC()]
        self.utterance_srcc, self.frame_srcc = [SRCC(), SRCC()]
        self.utterance_mse, self.frame_mse = [MSE(), MSE()]

    def update(self, utterance_scores: torch.Tensor, frame_scores: List[torch.Tensor], mos_scores: torch.Tensor):
        '''
            adds batch values to the metrics
        '''
        for i in range(len(utterance_scores)):
            utterance_score = utterance_scores[i].float().item()
            mos_score = mos_scores[i].float().item()

            self.utterance_lcc.add_value(utterance_score, mos_score)
            self.utterance_srcc.add_value(utterance_score, mos_score)
            self.utterance_mse.add_value(utterance_score, mos_score)

            for j in range(len(frame_scores[i])):
                frame_score = frame_scores[i][j].float().item()

                self.frame_lcc.add_value(frame_score, mos_score)
                self.frame_srcc.add_value(frame_score, mos_score)
                self.frame_mse.add_value(frame_score, mos_score)

    def print(self, epoch_num: int = None):
        print(f'{datetime.now()} :: {self.name.upper()}'+(f' :: EPOCH {epoch_num}' if epoch_num else ''))
        print('    UTTERANCE LCC: ', self.utterance_lcc.calculate())
        print('    UTTERANCE SRCC: ', self.utterance_srcc.calculate())
        print('    UTTERANCE MSE: ', self.utterance_mse.calculate())
        print('    FRAME LCC: ', self.frame_lcc.calculate())
        print('    FRAME SRCC: ', self.frame_srcc.calculate())
        print('    FRAME MSE: ', self.frame_mse.calculate())

    def save(self):
        '''
            appends metrics to a file, named by the title, in the runs directory specified in the environment variable
            clears all metrics after saving
        '''
        pass