import os
import librosa
import torch
import pandas as pd
from typing import List, Optional
from torch.utils.data import Dataset


def encode_text(text):
    '''
        numbers 48-57
        uppercase letters 65-90
        lowercase letters 97-122
    '''
    text_char_codes = [ ord(char) for char in text ]
    return [
        char_code
        for char_code in text_char_codes
        if char_code in range(48, 58) or char_code in range(65, 91) or char_code in range(97, 123)
    ]


def encode_system_id(dataset, system_id):
    return dataset.system_ids\
        .loc[dataset.system_ids.system_id == system_id]\
        .index[0]


class SOMOSMeanTrainDataset(Dataset):
    def __init__(self, 
                 sample_rate: int, 
                 data_dir: str = 'data',
                 system_ids: Optional[List[str]] = None,
                 ):

        self.sample_rate = sample_rate
        self.data_dir = data_dir

        scores_path = os.path.join(self.data_dir, 'SOMOS/training_files/split1/clean/train_mos_list.txt')
        # expects transcripts to be loaded via gather_transcripts.py and saved in SOMOS directory
        transcripts_path = os.path.join(self.data_dir, 'SOMOS/all_transcripts.txt')

        scores = pd.read_csv(scores_path)
        scores['wav_id'] = scores.utteranceId.str.split('.').str[0]
        scores['system_id'] = scores.wav_id.str.split('_').str[-1]
        transcripts = pd.read_csv(transcripts_path, delimiter='\t', names=['wav_id', 'text'])

        self.data = scores.merge(transcripts, on='wav_id')

        # select only scores from particular systems if they are included in the system_ids argument (for testing)
        if system_ids:
            self.data = self.data[self.data.system_id.isin(system_ids)]

        self.system_ids = self.data.system_id\
            .value_counts()\
            .reset_index()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        audio_path = os.path.join(self.data_dir, 'SOMOS/audios', sample.utteranceId)
        audio, sr = librosa.load(audio_path, sr=None)
        # downsample/upsample audio to sample_rate
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        text = encode_text(sample.text)
        system_id = encode_system_id(self, sample.system_id)
        return torch.tensor(audio),\
            torch.tensor(text),\
            torch.tensor(system_id),\
            torch.tensor(sample['mean'])


class SOMOSTestValDataset(Dataset):
    def __init__(self,
                 sample_rate: int,
                 data_path: str,
                 transcripts_path: str = 'data/SOMOS/all_transcripts.txt',
                 audios_path: str = 'data/SOMOS/audios',
                 ):

        self.sample_rate = sample_rate
        self.data_path = data_path
        self.transcripts_path = transcripts_path
        self.audios_path = audios_path

        scores = pd.read_csv(self.data_path)
        scores['wav_id'] = scores.utteranceId.str.split('.').str[0]
        scores['system_id'] = scores.wav_id.str.split('_').str[-1]

        transcripts = pd.read_csv(self.transcripts_path, delimiter='\t', names=['wav_id', 'text'])

        self.data = scores.merge(transcripts, on='wav_id')

        # encode listener ids
        self.data.listenerId = pd.factorize(self.data.listenerId)[0]

        self.system_ids = self.data.system_id\
            .value_counts()\
            .reset_index()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        audio, sr = librosa.load(os.path.join(self.audios_path, sample.utteranceId), sr=None)
        # downsample/upsample audio to sample_rate
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        text = encode_text(sample.text)
        system_id = encode_system_id(self, sample.system_id)
        return torch.tensor(audio),\
            torch.tensor(text),\
            torch.tensor(system_id),\
            torch.tensor(sample.listenerId),\
            torch.tensor(sample.choice, dtype=torch.float32)
