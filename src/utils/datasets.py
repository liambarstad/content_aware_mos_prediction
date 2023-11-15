import os
import librosa
import torch
import scipy
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from torch.utils.data import Dataset

class SOMOSDataset(Dataset):
    def __init__(self, 
                 stft_params: Dict[str, int],
                 sample_rate: int = 22050, 
                 system_ids: Optional[List[str]] = None,
                 locales: Optional[List[str]] = None,
                 split: Optional[str] = None,
                 reliability_percentile: float = None,
                ):
        self.stft_params = stft_params
        self.sample_rate = sample_rate

        self.scores_dir = \
            os.environ['SOMOS_COMBINED_SCORES_LOC'] if not split \
            else os.environ['SOMOS_TRAIN_SCORES_LOC'] if split == 'train' \
            else os.environ['SOMOS_VAL_SCORES_LOC'] if split == 'validation' \
            else os.environ['SOMOS_TEST_SCORES_LOC'] if split == 'test' \
            else None
        if not self.scores_dir:
            raise ValueError('Invalid split argument, must be in [train, validation, test] or None')

        self.audio_dir = os.environ['SOMOS_AUDIO_DIR']
        
        self.scores = pd.read_csv(self.scores_dir, delimiter=r'\s+', engine='python')

        # select only scores from particular systems if they are included in the system_ids argument (for testing)
        if system_ids:
            self.scores = self.scores[self.scores.systemId.isin(system_ids)]

        # select only particular locales if they are included in the locales argument
        if locales:
            self.scores = self.scores[self.scores.locale.isin(locales)]

        # select only scores from reliable listeners within a given percentile if specified
        if reliability_percentile:
            percentile_value = self.scores.listenerReliability.quantile(reliability_percentile)
            self.scores = self.scores[self.scores.listenerReliability >= percentile_value]
        
    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        row = self.scores.iloc[idx]
        audio_fname = os.path.join(self.audio_dir, row.utteranceId)+'.wav'
        audio, sr = librosa.load(audio_fname, sr=None)
        mels = self._prepare_audio(audio, sr)
        return torch.tensor(mels), torch.tensor(row.choice), row.to_dict()

    def _prepare_audio(self, audio, orig_sr):
        audio = self._downsample(audio, orig_sr)
        # convert to stft, take absolute value to remove complex numbers
        return np.transpose(np.abs(librosa.stft(
            y=audio, 
            **self.stft_params,
            window=scipy.signal.windows.hamming
        )))

    def _downsample(self, audio, orig_sr):
        # downsample if sample rate specified in self.sample_rate
        if self.sample_rate:
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
        else:
            return audio
