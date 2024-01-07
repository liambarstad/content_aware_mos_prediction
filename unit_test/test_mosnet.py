import pytest
import torch
from torch.utils.data import DataLoader

from ..src.datasets import SOMOSMeanTrainDataset
from ..src.mosnet.models import MOSNet
from ..src.utils import pad_audio_batch

class TestMOSNet:
    def setup_method(self):
        self.stft_params = {
            'n_fft': 512,
            'hop_length': 256
        }
        self.model_params = {
            'cnn_channels': [16, 32, 64, 128],
            'cnn_kernel_size': 3,
            'blstm_hidden_size': 128,
            'fc_hidden_size': 128,
            'fc_dropout': 0.3
        }
        self.dataset = SOMOSMeanTrainDataset(
            sample_rate=16000,
            data_dir='data'
        )
        self.dataloader = DataLoader(
            self.dataset, 
            6, 
            collate_fn=pad_audio_batch,
            shuffle=True  
        )

    def test_will_return_mos_score(self):
        audio, audio_lengths, _, _, _, mos_score = next(iter(self.dataloader))
        mosnet = MOSNet(
            stft_params=self.stft_params,
            model_params=self.model_params
        )
        mosnet.train()
        utterance_mos, frame_mos = mosnet(
            audio=audio, 
            audio_lengths=audio_lengths, 
            foo='bar'
        ) 
        assert type(utterance_mos) == torch.Tensor
        assert type(frame_mos) == list
        assert type(frame_mos[0]) == torch.Tensor
        # no outputs are above 5 or below 1
        assert (torch.sum(frame_mos[0] > 5) == 0).item()
        assert (torch.sum(utterance_mos > 5) == 0).item()
        assert (torch.sum(frame_mos[0] < 1) == 0).item()
        assert (torch.sum(utterance_mos < 1) == 0).item()

# 
'''
class TestMOSNet:

    def setup_method(self):
        self.default_args = {
            'cnn_channels': [16, 32, 64, 128],
            'cnn_kernel_size': 3,
            'blstm_hidden_size': 128,
            'fc_hidden_size': 128,
            'fc_dropout': 0.3
        }
        self.dataset = SOMOSDataset(stft_params={
            'hop_length': 256,
            'n_fft': 512
        }, sample_rate=16000, split='test')
        self.dataloader = DataLoader(self.dataset, 6, collate_fn=pad_audio_batch)

    def test_will_return_mos_score(self):
        audio, sample_lengths, mos_score, _ = next(iter(self.dataloader))
        mosnet = MOSNet(**self.default_args)
        utterance_mos, frame_mos = mosnet(audio, sample_lengths) 
        assert type(utterance_mos) == torch.Tensor
        assert type(frame_mos) == list
        assert type(frame_mos[0]) == torch.Tensor
        # no outputs are above 5 or below 1
        assert (torch.sum(frame_mos[0] > 5) == 0).item()
        assert (torch.sum(utterance_mos > 5) == 0).item()
        assert (torch.sum(frame_mos[0] < 1) == 0).item()
        assert (torch.sum(utterance_mos < 1) == 0).item()
'''