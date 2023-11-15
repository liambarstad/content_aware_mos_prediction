import os
import sys
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append('src')
from mosnet.model import MOSNet
from utils.params import Params
from utils.datasets import SOMOSDataset
from utils.loss import mosnet_loss
from utils.metrics import MOSNetMetrics
from utils.utils import pad_audio_batch

DEFAULT_DATASET_PARAMS = {
    'system_ids': [15],
    'sample_rate': 16000,
    #'locales': ['us'],
    #'reliability_percentile': 0.80
}

def validate(
        config_path: str, 
        mosnet: MOSNet, 
        epoch: int = None, 
        run_key: str = '1'
        ):
    '''
        validates a mosnet model, using the parameters specified in the config yaml file specified by config_path
        takes an optional epoch number to print and in the metrics, determining when the model was validated during training
        the run_key is used to differentiate between runs of the same model

        prints metrics every n epochs, and saves metrics to a file every epoch
        returns the validation loss
    '''

    params = Params(config_path)

    device = params.meta['device']

    somos_validation = SOMOSDataset(
        stft_params=params.stft,
        split='validation',
        **DEFAULT_DATASET_PARAMS
    )

    validation_loader = DataLoader(
        somos_validation,
        params.train['batch_size'],
        shuffle=True,
        collate_fn=pad_audio_batch,
        num_workers=params.train['num_workers']
    )

    val_metrics = MOSNetMetrics('mosnet_validation', run_key)

    mosnet.eval()
    with torch.no_grad():
        for i, (audio, sample_lengths, mos_scores, metadata) in enumerate(validation_loader):
            utterance_scores, frame_scores = mosnet(audio.to(device), sample_lengths) 
            val_loss = mosnet_loss(utterance_scores, frame_scores, mos_scores.to(device), params.train['frame_weighting_factor'])
            val_metrics.update(utterance_scores, frame_scores, mos_scores)

    val_metrics.print(epoch)
    val_metrics.save(epoch)

    return val_loss.item()


def train(
        config_path: str, 
        validate_loop: bool = True, 
        early_stopping_delta: float = 0.001,
        early_stopping_patience: int = 5,
        early_stopping_min_epochs: int = 10,
        seed_value: int = 123, 
        run_key: str = '1'
        ):
    '''
        trains the mosnet model, using the parameters specified in the config yaml file specified by config_path
        if validate_loop is True, will validate every n epochs, where n is specified in the config yaml file
        implements early stopping, if validation loss does not improve by early_stopping_delta for early_stopping_patience epochs
        the seed_value is used to seed the random number generators for reproducibility
        the run_key is used to differentiate between runs of the same model

        prints metrics every n epochs, and saves metrics to a file every epoch
        saves model to mosnet.pth in the directory specified by the environment variable SAVED_MODEL_DIR
        returns the trained mosnet model
    '''

    params = Params(config_path)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)

    device = params.meta['device']

    somos_train = SOMOSDataset(
        stft_params=params.stft,
        split='train',
        **DEFAULT_DATASET_PARAMS
    )

    train_loader = DataLoader(
        somos_train,
        params.train['batch_size'],
        shuffle=True,
        collate_fn=pad_audio_batch,
        num_workers=params.train['num_workers']
    )

    mosnet = MOSNet(**params.model).to(device)

    optimizer = torch.optim.Adam(
        mosnet.parameters(),
        lr=params.train['learning_rate']
    )
    
    # validation losses for early stopping
    validation_losses = []

    print(f'{datetime.now()} :: START RUN')
    for epoch in range(params.train['epochs']):
        train_metrics = MOSNetMetrics('mosnet_train', run_key) 

        mosnet.train()
        for i, (audio, sample_lengths, mos_scores, metadata) in enumerate(train_loader):
            optimizer.zero_grad()
            
            utterance_scores, frame_scores = mosnet(audio.to(device), sample_lengths) 

            loss = mosnet_loss(utterance_scores, frame_scores, mos_scores.to(device), params.train['frame_weighting_factor'])
            loss.backward()
            optimizer.step()

            train_metrics.update(utterance_scores, frame_scores, mos_scores) 

        # validation loop every n epochs
        if epoch == 0 or (epoch+1) % params.train['validate_every_n_epochs'] == 0:
            # print metrics every n epochs regardless
            train_metrics.print(epoch+1)

            if validate_loop:
                val_loss = validate(config_path, mosnet, epoch+1, run_key)
                validation_losses.append(val_loss)
                # stop early if validation loss does not improve by early_stopping_delta for early_stopping_patience epochs
                # does not stop early until early_stopping_min_epochs epochs have passed
                if epoch >= early_stopping_min_epochs and \
                    len(validation_losses) > early_stopping_patience+1 and all(
                        l < validation_losses[-early_stopping_patience-1] + early_stopping_delta 
                        for l in validation_losses[-early_stopping_patience:]
                    ):
                    print(f'{datetime.now()} :: EARLY STOPPING AT EPOCH {epoch+1}')

                    return mosnet
        else:
            print(f'{datetime.now()} :: EPOCH {epoch+1}')

        train_metrics.save(epoch+1) 

    torch.save(mosnet, os.path.join(os.environ['SAVED_MODEL_DIR'], 'mosnet.pth'))

    return mosnet


if __name__ == '__main__':
    train('src/mosnet/config.yml', validate_loop=False)