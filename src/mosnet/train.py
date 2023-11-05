import sys
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import MOSNet
sys.path.append('src')
from utils.params import Params
from utils.datasets import SOMOSDataset
from utils.loss import mosnet_loss
from utils.metrics import MOSNetMetrics
from utils.utils import pad_audio_batch

DEFAULT_DATASET_PARAMS = {
    'sample_rate': 16000,
    'locales': ['us'],
    'reliability_percentile': 0.80
}

def validate(config_path: str, mosnet: MOSNet, epoch: int = None):
    params = Params(config_path)

    device = params.meta['device']

    somos_validation = SOMOSDataset(
        mel_params=params.mel,
        split='validation',
        **DEFAULT_DATASET_PARAMS
    )

    validation_loader = DataLoader(
        somos_validation,
        params.train['batch_size'],
        shuffle=True,
        collate_fn=pad_audio_batch,
        #num_workers=params.train['num_workers']
    )

    val_metrics = MOSNetMetrics('mosnet_validation')

    mosnet.eval()
    with torch.no_grad():
        for i, (audio, sample_lengths, mos_scores, metadata) in enumerate(validation_loader):
            utterance_scores, frame_scores = mosnet(audio.to(device), sample_lengths) 
            
            val_metrics.update(utterance_scores, frame_scores, mos_scores)

    val_metrics.print(epoch)
    val_metrics.save()


def train(config_path: str, validate_loop: bool = True, seed_value: int = 123):
    params = Params(config_path)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)

    device = params.meta['device']

    somos_train = SOMOSDataset(
        mel_params=params.mel,
        split='train',
        **DEFAULT_DATASET_PARAMS
    )

    train_loader = DataLoader(
        somos_train,
        params.train['batch_size'],
        shuffle=True,
        collate_fn=pad_audio_batch,
        #num_workers=params.train['num_workers']
    )

    mosnet = MOSNet(**params.model).to(device)

    optimizer = torch.optim.Adam(
        mosnet.parameters(),
        lr=params.train['learning_rate']
    )
    
    for epoch in range(params.train['epochs']):
        train_metrics = MOSNetMetrics('mosnet_train') 

        print(f'{datetime.now()} :: EPOCH {epoch+1}')
        mosnet.train()
        for i, (audio, sample_lengths, mos_scores, metadata) in enumerate(train_loader):
            optimizer.zero_grad()
            
            utterance_scores, frame_scores = mosnet(audio.to(device), sample_lengths) 

            loss = mosnet_loss(utterance_scores, frame_scores, mos_scores)
            loss.backward()

            optimizer.step()

            train_metrics.update(utterance_scores, frame_scores, mos_scores)            

        train_metrics.save() 

        # validation loop every n epochs
        if epoch % params.train['validate_every_n_epochs']:
            # print metrics every n epochs regardless
            train_metrics.print(epoch+1)

            if validate_loop:
                validate(config_path, mosnet, epoch+1)

    return mosnet


if __name__ == '__main__':
    train('mosnet/config.yml')