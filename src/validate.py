import os
import torch
from torch.utils.data import DataLoader

from .datasets import SOMOSTestValDataset
from .metrics import Metrics

def validate(
    model,
    data_path: str,
    num_workers: int = 0,
    transcripts_path: str ='data/SOMOS/all_transcripts.txt',
    audios_path: str ='data/SOMOS/audios',
    device: str = 'cpu',
    runs_dir: str = 'runs'
):
    somos_val = SOMOSTestValDataset(
        sample_rate=16000,
        data_path=data_path,
        transcripts_path=transcripts_path,
        audios_path=audios_path
    )
    
    val_loader = DataLoader(
        somos_val,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers
    )

    val_metrics = Metrics('mosnet validation')
    val_loss = 0.0

    model.eval()

    with torch.no_grad():
        for audio, text, system_ids, _, mos_scores in val_loader:

            utterance_scores, frame_scores = model(
                audio=audio.to(device),
                text=text.to(device)
            )

            val_metrics.update(utterance_scores, system_ids, mos_scores.to(device))

            val_loss += model.loss(
                utterance_scores=utterance_scores, 
                frame_scores=frame_scores, 
                mos_scores=mos_scores.to(device)
            ).item()

    val_metrics.print()
    val_metrics.save(os.path.join(runs_dir, 'validation.csv'))

    return val_loss


