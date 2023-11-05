from typing import List
import torch
from torch import nn


def mosnet_loss(
        utterance_scores: torch.Tensor, 
        frame_scores: List[torch.Tensor], 
        mos_scores: torch.Tensor, 
        weighting_factor: int = 1
        ):
    '''
        loss function for mosnet models
        adds the utterance/target MSE to the MSE per frame for each utterance (weighted by the weighting_factor)
    '''

    mos_scores = mos_scores.float()
    mse_loss = nn.MSELoss()

    utterance_loss = mse_loss(utterance_scores, mos_scores)

    frame_losses = [
        mse_loss(sc, mos_scores[i].expand_as(sc))
        for i, sc in enumerate(frame_scores)
    ]

    frame_loss = torch.mean(torch.stack(frame_losses))

    return utterance_loss + (weighting_factor * frame_loss)
