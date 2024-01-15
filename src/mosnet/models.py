import sys
import torch
from typing import List, Dict
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import Spectrogram

from ..utils import conv_same_padding


class MOSCNN(nn.Module):
    def __init__(self,
                 cnn_channels: List[int],
                 cnn_kernel_size: int
                 ):

        super(MOSCNN, self).__init__()
        self.cnn_kernel_size = cnn_kernel_size

        # for each element (x) in cnn_channels, adds 3 convolutional layers with x channels
        self.layers = nn.ModuleList([
            nn.Sequential(
                self.conv_layer(
                    in_channels=(1 if ind == 0 else cnn_channels[ind-1]),
                    out_channels=channels
                ),
                nn.ReLU(),
                self.conv_layer(
                    in_channels=channels,
                    out_channels=channels
                ),
                nn.ReLU(),
                self.conv_layer(
                    in_channels=channels,
                    out_channels=channels,
                    stride=(1, 3)
                ),
                nn.ReLU(),
            )
            for ind, channels in enumerate(cnn_channels)
        ])

    def conv_layer(self, in_channels: int, out_channels: int, stride: int = 1):
        # adds initialization function for conv layer and maintains same padding
        conv_layer = nn.Conv2d(
             in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=(self.cnn_kernel_size, self.cnn_kernel_size),
             stride=stride,
             padding=conv_same_padding(self.cnn_kernel_size)
        )
        nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
        return conv_layer

    def forward(self, x: torch.Tensor):
        #accepts a tensor [bz, frame_size, mel], assuming one channel
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return x


class MOSNet(nn.Module):
    def __init__(self, stft_params: Dict, model_params: Dict):
        super(MOSNet, self).__init__()

        self.cnn_channels = model_params['cnn_channels']
        self.cnn_kernel_size = model_params['cnn_kernel_size']
        self.blstm_hidden_size = model_params['blstm_hidden_size']
        self.fc_hidden_size = model_params['fc_hidden_size']
        self.fc_dropout = model_params['fc_dropout']

        self.stft = Spectrogram(**stft_params) 

        self.cnn = MOSCNN(self.cnn_channels, self.cnn_kernel_size)
        
        self.blstm = nn.LSTM(
            input_size=self.cnn_channels[-1]*4,
            hidden_size=self.blstm_hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Linear(
            in_features=2*self.blstm_hidden_size,
            out_features=self.fc_hidden_size
        )

        self.frame_score = nn.Linear(
            in_features=self.fc_hidden_size,
            out_features=1
        )

    def forward(self, audio: torch.Tensor, audio_lengths: torch.Tensor = None, **kwargs):
        '''
            expects audio to be in shape [bz, ts, sample_points]
            sample_lengths is a list of the original lengths of each sample in the batch, to prevent the network from running on empty frames
            returns 2 items, a tensor of scalar utterance-level predictions, and a list of frame-level predictions of shape [n_frames]
        '''
        freqs = self.stft(audio).transpose(1, 2)
        cnn_output = self.cnn(freqs) 
        # transpose the cnn output so that the channels are now the features, and flatten the height dimension
        blstm_input = torch.transpose(cnn_output, 1, 2).flatten(2, 3)
        blstm_output, _ = self.blstm(blstm_input)
        fc1_output = F.relu(F.dropout(self.fc1(blstm_output), p=self.fc_dropout, training=self.training))
        # activation layer and constrain to be in range [1, 5]
        frame_mos = torch.sigmoid(self.frame_score(fc1_output)) * (5 - 1) + 1
        if audio_lengths is not None:
            # remove padding by using sample_lengths
            frame_predictions = [ sample[:audio_lengths[ind]].squeeze(1) for ind, sample in enumerate(frame_mos) ] 
        else:
            frame_predictions = [ sample.squeeze(1) for sample in frame_mos ]
        utterance_predictions = torch.stack([ torch.mean(sample) for sample in frame_predictions ])

        return utterance_predictions, frame_predictions

    def loss(
        self,
        utterance_scores: torch.Tensor, 
        frame_scores: List[torch.Tensor], 
        mos_scores: torch.Tensor, 
        weighting_factor: int = 1,
        **kwargs
        ):
        '''
            loss function for mosnet models
            adds the utterance/target MSE to the MSE per frame for each utterance (weighted by the weighting_factor)
        '''
        mos_scores = mos_scores.float()
        # expand targets to match frame_predictions shape
        expanded_mos = [
            mos_score.expand(len(frame)) 
            for mos_score, frame in zip(mos_scores, frame_scores)
        ]

        # calculate frame-level loss
        frame_loss = F.mse_loss(torch.cat(frame_scores), torch.cat(expanded_mos))

        # calculate utterance-level loss
        utterance_loss = F.mse_loss(utterance_scores, mos_scores)

        # can adjust the weights here depending on how much importance you want to give to each loss
        return utterance_loss + (weighting_factor * frame_loss)


class ProsAlignMOSNet(MOSNet):
    pass