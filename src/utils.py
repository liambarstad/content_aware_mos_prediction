import torch
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence


def log(message, level='INFO'):
    print(f'{datetime.now()} :: {message}')

def pad_audio_batch(batch):
    '''
        for use in the torch DataLoader class as the collate_fn, 
        assumes that the batch is formatted [audio, text, system_id, mos_score]
        returns audio lengths and text lengths in addition to the padded audio and text
    '''
    audio, text, system_id, mos_scores = zip(*batch)
    # pad audio
    audio_lengths = [ sample.shape[0] for sample in audio ]
    audio = pad_sequence(audio, batch_first=True)
    # pad text
    text_lengths = [ sample.shape[0] for sample in text ]
    text = pad_sequence(text, batch_first=True)
    # pass audio lengths and text lengths to training loop
    return audio,\
        torch.tensor(audio_lengths),\
        text,\
        torch.tensor(text_lengths),\
        torch.tensor(system_id),\
        torch.tensor(mos_scores)

def conv_same_padding(kernel_size=1, stride=1, dilation=1):
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2