import torch
from torch.utils.data import DataLoader

from .datasets import SOMOSTestValDataset
from .utils import pad_audio_batch
from .metrics import Metrics

def test(
    model,
    run_name: str,
    model_name: str,
    data_path: str,
    batch_size: int = 64,
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
        batch_size,
        shuffle=True,
        collate_fn=pad_audio_batch,
        num_workers=num_workers
    )

    val_metrics = Metrics(run_name, model_name)
    val_loss = 0.0

    model.eval()

    with torch.no_grad():
        for i, (audio, audio_lengths, text, text_lengths, system_ids, mos_scores) in enumerate(val_loader):

            utterance_scores, frame_scores = model(
                audio=audio.to(device),
                audio_lengths=audio_lengths.to(device),
                text=text.to(device),
                text_lengths=text_lengths.to(device)
            )

            val_metrics.update(utterance_scores, system_ids, mos_scores.to(device))

            val_loss += model.loss(
                utterance_scores=utterance_scores, 
                frame_scores=frame_scores, 
                mos_scores=mos_scores.to(device)
            ).item()

    val_metrics.print()
    val_metrics.save(runs_dir)

    return val_loss


