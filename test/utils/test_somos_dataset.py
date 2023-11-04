import sys
import os
import pytest
import torch

sys.path.append('src')
from utils.datasets import SOMOSDataset

class TestSOMOSDataset:

    def setup_method(self):
        self.default_args = {
            'sample_rate': 16000,
            'mel_params': {
                'n_fft': 512,
                'hop_length': 256,
                'n_mels': 80
            }
        }

    def test_will_load_all_data(self):
        all_data = SOMOSDataset(**self.default_args)
        assert len(all_data) == 374955
        audio, mos_score, info = all_data[0]
        assert type(audio) == torch.Tensor
        assert audio.shape[1] == self.default_args['mel_params']['n_mels']
        assert type(mos_score) == torch.Tensor
        assert mos_score <= 5 and mos_score >= 0

    def test_will_load_train_data(self):
        train_data = SOMOSDataset(**self.default_args, split='train')
        assert len(train_data) == 251635
        audio, mos_score, info = train_data[0]
        assert type(audio) == torch.Tensor
        assert audio.shape[1] == self.default_args['mel_params']['n_mels']
        assert type(mos_score) == torch.Tensor
        assert mos_score <= 5 and mos_score >= 0

    def test_will_load_validation_data(self):
        validation_data = SOMOSDataset(**self.default_args, split='validation')
        assert len(validation_data) == 53569
        audio, mos_score, info = validation_data[0]
        assert type(audio) == torch.Tensor
        assert audio.shape[1] == self.default_args['mel_params']['n_mels']
        assert type(mos_score) == torch.Tensor
        assert mos_score <= 5 and mos_score >= 0

    def test_will_load_test_data(self):
        test_data = SOMOSDataset(**self.default_args, split='test')
        assert len(test_data) == 53896
        audio, mos_score, info = test_data[0]
        assert type(audio) == torch.Tensor
        assert audio.shape[1] == self.default_args['mel_params']['n_mels']
        assert type(mos_score) == torch.Tensor
        assert mos_score <= 5 and mos_score >= 0

    def test_can_filter_by_locale(self):
        pass

