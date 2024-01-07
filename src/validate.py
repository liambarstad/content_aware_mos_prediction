#TEST_VAL_PARAMS = {
#    'listeners': 10,
#    'systems': 10,
#    'texts': 50
#}
import torch
from torch.utils.data import DataLoader

from .datasets import SOMOSTestValDataset
from .utils import pad_audio_batch
from .metrics import Metrics

def validate(
    model,
    run_name,
    model_name,
    batch_size,
    num_workers,
    data_dir: str = 'data',
    listener_count: int = 10,
    system_count: int = 10,
    text_count: int = 50
):
    somos_val = SOMOSTestValDataset(
        split='validation',
        sample_rate=16000,
        listener_count=listener_count,
        system_count=system_count,
        text_count=text_count,
        data_dir=data_dir 
    )
    
    val_loader = DataLoader(
        somos_val,
        batch_size,
        shuffle=True,
        collate_fn=pad_audio_batch,
        num_workers=num_workers
    )

    val_metrics = Metrics(run_name, model_name)

    model.eval()
    import ipdb; ipdb.sset_trace()

    with torch.no_grad():
        for i, (audio, audio_lengths, text, text_lengths, system_ids, mos_scores) in enumerate(val_loader):

            utterance_scores, frame_scores = model(
                audio=audio.to(device),
                audio_lengths=audio_lengths.to(device),
                text=text.to(device),
                text_lengths=text_lengths.to(device)
            )
            utterance_scores, frame_scores = mosnet(audio.to(device), sample_lengths) 
            val_loss = mosnet_loss(utterance_scores, frame_scores, mos_scores.to(device), params.train['frame_weighting_factor'])
            val_metrics.update(utterance_scores, frame_scores, mos_scores)

    val_metrics.print(epoch)
    val_metrics.save(epoch)

    return val_loss.item()


