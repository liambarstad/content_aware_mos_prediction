import os
import csv
import torch
from typing import List
from datetime import datetime
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


class Metric:
    '''
        base class for metrics, creates a list of predictions and targets 
        "calculate" function can then be overwritten to calculate the metric based on the predictions and targets
    '''
    def __init__(self, name: str):
        self.name = name
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

class KTAU(Metric):
    '''
        kendall's tau
    '''
    def calculate(self):
        if len(self.predictions) < 3:
            return None
        return kendalltau(self.predictions, self.targets)[0]

class LCC(Metric):
    '''
        linear correlation coefficient
    '''
    def calculate(self):
        if len(self.predictions) < 3:
            return None
        return pearsonr(self.predictions, self.targets)[0]


class SRCC(Metric):
    '''
        spearman rank correlation coefficient
    '''
    def calculate(self):
        if len(self.predictions) < 3:
            return None
        return spearmanr(self.predictions, self.targets)[0]


class MSE(Metric):
    '''
        mean squared error
    '''
    def calculate(self):
        return ((np.array(self.predictions) - np.array(self.targets)) ** 2).mean()


class Metrics:
    def __init__(self, name: str, runs_dir: str = '/', run_key: str = '', model_name: str = ''):
        '''
            calculates all metrics, accepts a "name" to differentiate between models and train/val/test, and a run_key to differentiate between runs
        '''
        self.name = name
        self.run_key = run_key
        self.utterance_lccs = LCC('utterace_lcc')
        self.utterance_srccs = SRCC('utterance_srcc')
        self.utterance_mses = MSE('utterance_mse')
        self.utterance_ktaus = KTAU('utterance_ktau')
    
        self.system_metrics = {}

    def update(self, 
               utterance_scores: torch.Tensor, 
               system_ids: torch.Tensor, 
               mos_scores: torch.Tensor
               ):
        '''
            adds batch values to the metrics
        '''
        for i in range(len(utterance_scores)):
            utterance_score = utterance_scores[i].float().item()
            mos_score = mos_scores[i].float().item()

            self.utterance_lccs.add_value(utterance_score, mos_score)
            self.utterance_srccs.add_value(utterance_score, mos_score)
            self.utterance_mses.add_value(utterance_score, mos_score)
            self.utterance_ktaus.add_value(utterance_score, mos_score)

        for system_id in system_ids:
            sys_val = system_id.item()
            mos_score = mos_scores[i].float().item()
            if not sys_val in self.system_metrics:
                self.system_metrics[sys_val] = {
                    'lcc': LCC(f'system_{sys_val}_lcc'),
                    'srcc': SRCC(f'system_{sys_val}_srcc'),
                    'mse': MSE(f'system_{sys_val}_mse'),
                    'ktau': KTAU(f'system_{sys_val}_ktau')
                }
            self.system_metrics[sys_val]['lcc'].add_value(utterance_score, mos_score)
            self.system_metrics[sys_val]['srcc'].add_value(utterance_score, mos_score)
            self.system_metrics[sys_val]['mse'].add_value(utterance_score, mos_score)
            self.system_metrics[sys_val]['ktau'].add_value(utterance_score, mos_score)

    def print(self, epoch_num: int = None):
        metrics = self.calculate()
        epoch_str = f':: EPOCH {epoch_num} ' if epoch_num else ''
        print(f'{datetime.now()} {epoch_str}:: {self.name.upper()} :: METRICS :: ')
        metrics_str = ''
        metrics_str += f'UTTERANCE :: {{ lcc: {metrics[0]:.4f}, srcc: {metrics[1]:.4f}, mse: {metrics[2]:.4f}, ktau: {metrics[3]:.4f} }} :: '
        metrics_str += f'SYSTEM :: {{ lcc: {metrics[4]:.4f}, srcc: {metrics[5]:.4f}, mse: {metrics[6]:.4f}, ktau: {metrics[7]:.4f} }}'
        print(metrics_str)

    def calculate(self):
        utterance_lcc = self.utterance_lccs.calculate()
        utterance_srcc = self.utterance_srccs.calculate()
        utterance_mse = self.utterance_mses.calculate()
        utterance_ktau = self.utterance_ktaus.calculate()
        
        return utterance_lcc,\
            utterance_srcc,\
            utterance_mse,\
            utterance_ktau,\
            self._calculate_system_metric('lcc'),\
            self._calculate_system_metric('srcc'),\
            self._calculate_system_metric('mse'),\
            self._calculate_system_metric('ktau'),\

    def _calculate_system_metric(self, metric: str):
        metric_values = []
        for system in self.system_metrics.values():
            metric_value = system[metric].calculate()
            if metric_value is not None:
                metric_values.append(metric_value)

        valid_metric_values = [v for v in metric_values if v is not None]
        return sum(metric_values) / len(valid_metric_values)
        
    def save(self, runs_dir, epoch_num: int = None):
        '''
            appends metrics to a file, named by the title and run_key, in the runs directory specified in the environment variable
            clears all metrics after saving
        '''
        run_fpath = os.path.join(runs_dir, self.name)+f'_{self.run_key}.csv'
        f_exists = os.path.isfile(run_fpath)
        os.makedirs(os.path.dirname(run_fpath), exist_ok=True)

        with open(run_fpath, 'a', newline='') as metrics_file:
            csvwriter = csv.writer(metrics_file)
            if not f_exists:
                csvwriter.writerow([
                    'epoch', 
                    'utterance_lcc',
                    'utterance_srcc', 
                    'utterance_mse', 
                    'utterance_ktau',
                    'system_lcc',
                    'system_srcc',
                    'system_mse',
                    'system_ktau'
                ])
            else: 
                csvwriter.writerow([epoch_num, *self.calculate()])
            
    def clear(self):
        '''
            clears all metrics
        '''

        for metric in [
            self.utterance_lccs, 
            self.utterance_srccs, 
            self.utterance_mses, 
            self.utterance_ktaus,
            ]:
            metric.clear()
        self.system_metrics = {}
