from typing import List
import torch
import torch.nn.functional as F

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