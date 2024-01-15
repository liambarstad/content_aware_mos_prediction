import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from .datasets import SOMOSTestValDataset
from .utils import pad_audio_batch
from .metrics import Metrics

def test(
    model_path: str,
    data_path: str = 'data/somos_prepared/test_set.csv',
    transcripts_path: str ='data/SOMOS/all_transcripts.txt',
    audios_path: str ='data/SOMOS/audios',
    device: str = 'cpu',
    runs_dir: str = 'runs',
    num_workers: int = 0,
):
    somos_test = SOMOSTestValDataset(
        sample_rate=16000,
        data_path=data_path,
        transcripts_path=transcripts_path,
        audios_path=audios_path
    )
    
    test_loader = DataLoader(
        somos_test,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers
    )

    test_metrics = Metrics('mosnet test')
    
    model = torch.load(model_path).to(device)
    model.eval()
    results_df = pd.DataFrame(columns=['system_id', 'listener_id', 'mos_score', 'predicted_score'])

    with torch.no_grad():
        for audio, text, system_ids, listener_ids, mos_scores in test_loader:

            utterance_scores, _ = model(
                audio=audio.to(device),
                text=text.to(device)
            )

            new_row = {
                'system_id': system_ids.item(),
                'listener_id': listener_ids.item(),
                'mos_score': mos_scores.item(),
                'predicted_score': utterance_scores.item()
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            test_metrics.update(utterance_scores, system_ids, mos_scores.to(device))

    test_metrics.print()
    test_metrics.save(os.path.join(runs_dir, 'test.csv'))

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(results_df)


            