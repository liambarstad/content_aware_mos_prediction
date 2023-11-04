import torch
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence


def log(message, level='INFO'):
    print(f'{datetime.now()} :: {message}')

def pad_audio_batch(batch):
    '''
        for use in the torch DataLoader class as the collate_fn, 
        assumes that the batch is formatted [audio, mos_scores, metadata]
        also assumes that the audio is formatted [ts, mels]
        returns padded audio, original sample lengths (List[int]), mos scores, and row metadata
    '''
    audio, mos_scores, metadata = zip(*batch)
    sample_lengths = [ sample.shape[0] for sample in audio ]
    audio = pad_sequence(audio, batch_first=True)
    mos_scores = torch.tensor(mos_scores)
    return audio, sample_lengths, mos_scores, metadata

def conv_same_padding(kernel_size=1, stride=1, dilation=1):
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2