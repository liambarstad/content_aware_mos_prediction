import pytest 
import torch
import pandas as pd

from ..src.datasets import SOMOSTestValDataset

class TestSOMOSTestValDataset:

    def setup_method(self):
       self.default_args = {
           'sample_rate': 16000,
       }     
        
    def test_will_load_val_data(self):
        somos_val_dataset = SOMOSTestValDataset(
            data_path='data/somos_prepared/validation_set.csv',
            **self.default_args
        )
        test_len = len(pd.read_csv('data/somos_prepared/validation_set.csv'))
        assert len(somos_val_dataset) == test_len
        audio, transcript, system_id, listener_id, mos_score = somos_val_dataset[0]
        assert type(audio) == torch.Tensor
        assert audio.dtype == torch.float32
        assert type(transcript) == torch.Tensor
        assert transcript.dtype == torch.int64
        assert type(system_id) == torch.Tensor
        assert system_id.dtype == torch.int64
        assert type(listener_id) == torch.Tensor
        assert listener_id.dtype == torch.int64
        assert type(mos_score) == torch.Tensor
        assert mos_score.dtype == torch.float32

    def test_will_load_test_data(self):
        somos_test_dataset = SOMOSTestValDataset(
            data_path='data/somos_prepared/test_set.csv',
            **self.default_args
        )
        test_len = len(pd.read_csv('data/somos_prepared/test_set.csv'))
        assert len(somos_test_dataset) == test_len
        audio, transcript, system_id, listener_id, mos_score = somos_test_dataset[0]
        assert type(audio) == torch.Tensor
        assert audio.dtype == torch.float32
        assert type(transcript) == torch.Tensor
        assert transcript.dtype == torch.int64
        assert type(system_id) == torch.Tensor
        assert system_id.dtype == torch.int64
        assert type(listener_id) == torch.Tensor
        assert listener_id.dtype == torch.int64
        assert type(mos_score) == torch.Tensor
        assert mos_score.dtype == torch.float32
        