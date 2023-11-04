import sys
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.append('src')
from utils.datasets import SOMOSDataset
from mosnet.model import MOSNet
from utils.utils import pad_audio_batch

class TestMOSNet:

    def setup_method(self):
        self.default_args = {
            'n_frames': 25,
            'cnn_channels': [16, 32, 64, 128],
            'cnn_kernel_size': 3,
            'blstm_hidden_size': 128,
            'fc_hidden_size': 128
        }
        self.dataset = SOMOSDataset(mel_params={
            'n_mels': 80,
            'hop_length': 256,
            'n_fft': 512
        }, sample_rate=16000, split='test')
        self.dataloader = DataLoader(self.dataset, 6, collate_fn=pad_audio_batch)

    def test_will_return_mos_score(self):
        audio, sample_lengths, mos_score, _ = next(iter(self.dataloader))
        mosnet = MOSNet(**self.default_args)
        utterance_mos, frame_mos = mosnet(audio, sample_lengths) 
        assert type(utterance_mos) == list
        assert type(frame_mos) == list
        assert type(utterance_mos[0]) == torch.Tensor
        assert type(frame_mos[0]) == torch.Tensor
        # no outputs are above 5 or below 1
        assert (torch.sum(frame_mos[0] > 5) == 0).item()
        assert (torch.sum(utterance_mos[0] > 5) == 0).item()
        assert (torch.sum(frame_mos[0] < 1) == 0).item()
        assert (torch.sum(utterance_mos[0] < 1) == 0).item()

    