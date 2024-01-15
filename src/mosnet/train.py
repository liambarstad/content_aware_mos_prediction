import os
import sys
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict
from torch.utils.data import DataLoader

from .loss import mosnet_loss
from ..datasets import SOMOSMeanTrainDataset
from ..metrics import Metrics
from ..utils import pad_audio_batch
from ..validate import validate

def train_mosnet(
    model: torch.nn.Module,
    train_params: Dict,
    validation_params: Dict,
    device: str = 'cpu',
    runs_dir: str = 'runs',
    seed_value: int = 123,
    frame_weighting_factor: float = 1.0,
    validate_every_n_epochs: int = 1,
    model_output_path: str = 'pretrained/mosnet.pt'
    ):

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    
    somos_train = SOMOSMeanTrainDataset(
        sample_rate=16000,
        data_dir=train_params['data_dir']
    )

    train_loader = DataLoader(
        somos_train,
        train_params['batch_size'],
        shuffle=True,
        collate_fn=pad_audio_batch,
        num_workers=train_params['num_workers']
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params['learning_rate']
    )
    
    # validation losses for early stopping
    validation_losses = []

    print(f'{datetime.now()} :: START RUN')
    train_metrics = Metrics('mosnet train') 

    for epoch in range(train_params['epochs']):
        # train loop
        for i, (audio, audio_lengths, text, text_lengths, system_ids, mos_scores) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            
            utterance_scores, frame_scores = model(
                audio=audio.to(device),
                audio_lengths=audio_lengths.to(device),
                text=text.to(device),
                text_lengths=text_lengths.to(device)
            ) 

            loss = model.loss(utterance_scores, frame_scores, mos_scores.to(device), frame_weighting_factor)
            loss.backward()
            optimizer.step()

            # add batch metrics
            train_metrics.update(utterance_scores, system_ids, mos_scores) 

        # print epoch metrics
        train_metrics.print(epoch+1)
        # save epoch metrics
        train_metrics.save(os.path.join(runs_dir, 'train.csv'), epoch+1)
        train_metrics.clear()

        # validation loop every n epochs
        if epoch == 0 or (epoch+1) % validate_every_n_epochs == 0:

            val_loss = validate(
                model,
                runs_dir=runs_dir,
                device=device,
                num_workers=train_params['num_workers'],
                **validation_params
            )

            validation_losses.append(val_loss)
            '''
                # stop early if validation loss does not improve by early_stopping_delta for early_stopping_patience epochs
                # does not stop early until early_stopping_min_epochs epochs have passed
                if epoch >= early_stopping_min_epochs and \
                    len(validation_losses) > early_stopping_patience+1 and all(
                        l < validation_losses[-early_stopping_patience-1] + early_stopping_delta 
                        for l in validation_losses[-early_stopping_patience:]
                    ):
                    print(f'{datetime.now()} :: EARLY STOPPING AT EPOCH {epoch+1}')

                    return mosnet
            '''

    torch.save(model, model_output_path)

    return model
