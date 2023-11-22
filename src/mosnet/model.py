import sys
import torch
from typing import List
from torch import nn
from torch.nn import functional as F

sys.path.append('src')
from shared.mos_cnn import MOSCNN


class MOSNet(nn.Module):
    def __init__(self,
                 cnn_channels: List[int],
                 cnn_kernel_size: int,
                 blstm_hidden_size: int,
                 fc_hidden_size: int,
                 fc_dropout: float = 0.3,
                 mos_score_range: List[int] = [1, 5]
                 ):

        super(MOSNet, self).__init__()

        self.fc_dropout = fc_dropout
        self.mos_score_range = mos_score_range

        self.cnn = MOSCNN(cnn_channels, cnn_kernel_size)

        self.blstm = nn.LSTM(
            input_size=cnn_channels[-1]*4,
            hidden_size=blstm_hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc1 = nn.Linear(
            in_features=2*blstm_hidden_size,
            out_features=fc_hidden_size
        )

        self.frame_score = nn.Linear(
            in_features=fc_hidden_size,
            out_features=1
        )

    def forward(self, audio: torch.Tensor, sample_lengths: List[int]):
        '''
            expects audio to be in shape [bz, ts, sample_points]
            sample_lengths is a list of the original lengths of each sample in the batch, to prevent the network from running on empty frames
            returns 2 items, a tensor of scalar utterance-level predictions, and a list of frame-level predictions of shape [n_frames]
        '''
        cnn_output = self.cnn(audio) 
        # transpose the cnn output so that the channels are now the features, and flatten the height dimension
        blstm_input = torch.transpose(cnn_output, 1, 2).flatten(2, 3)
        # run through blstm
        blstm_output, _ = self.blstm(blstm_input)
        # pass through fully connected layer w/ relu + dropout
        fc1_output = F.relu(F.dropout(self.fc1(blstm_output), p=self.fc_dropout, training=self.training))
        # activation layer and constrain to be in range
        frame_mos = torch.sigmoid(self.frame_score(fc1_output))\
            * (self.mos_score_range[1] - self.mos_score_range[0]) + self.mos_score_range[0]
        # remove padding by using sample_lengths
        frame_predictions = [ sample[:sample_lengths[ind]].squeeze(1) for ind, sample in enumerate(frame_mos) ] 
        # average the frame predictions to get the utterance prediction
        utterance_predictions = torch.stack([ torch.mean(sample) for sample in frame_predictions ])
        print('utterance preds:', utterance_predictions)

        return utterance_predictions, frame_predictions

    '''
    def forward(self, audio: torch.Tensor, sample_lengths: List[int]):
        for sample_ind in range(audio.shape[0]):
            # for each sample in the batch, split into frames
            sample = audio[sample_ind, :sample_lengths[sample_ind], :]\
                .unsqueeze(0)
            frames = torch.split(sample, self.n_frames, dim=1)

            for frame in frames:
                frame_mos = self._frame_mos(frame)
                # add the frame-level predictions to the list of predictions
                frame_predictions[sample_ind].append(frame_mos)

            frame_predictions[sample_ind] = torch.stack(frame_predictions[sample_ind]).view(-1)
            utterance_predictions.append(torch.mean(frame_predictions[sample_ind]))
            
        return torch.stack(utterance_predictions), frame_predictions

    def _frame_mos(self, frame: torch.Tensor):
        cnn_output = self.cnn(frame) 
        # transpose the cnn output so that the channels are now the features
        blstm_input = torch.transpose(cnn_output, 1, 2)
        # flatten channels dimension
        blstm_input = blstm_input.reshape(1, -1, blstm_input.shape[2]*blstm_input.shape[3])
        # run through blstm
        blstm_output, _ = self.blstm(blstm_input)
        # flatten the blstm output
        blstm_output_flattened = blstm_output.view(1, -1)
        # pad the output, in the case of final frame
        padding_len = self.fc1_input_dim - blstm_output_flattened.shape[1]
        blstm_output_flattened = F.pad(blstm_output_flattened, (0, padding_len), value=0)

        fc1_output = self.fc1(blstm_output_flattened)
        fc1_output = F.relu(F.dropout(fc1_output, p=self.fc_dropout, training=self.training))

        frame_mos = self.frame_score(fc1_output)
        '''
