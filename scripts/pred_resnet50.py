import sys
from pathlib import Path
import fire

import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path.cwd().parents[0]))
from conservision.dataset import ImagesDataset
from conservision.models import resnet50_animal
from utils.transforms import BASIC_TRANSFORM


def main(
        model_path: str,
        features_csv: str,
        images_root: str,
        prediction_path: str,
        batch_size: int = 256,
        device_name: str = 'cuda'
    ) -> None:

    device = torch.device(device_name)

    features = pd.read_csv(features_csv, index_col="id")
    features['filepath'] = features['filepath'].apply(lambda path: Path(images_root) / str(path)) # type: ignore

    x = features.filepath.to_frame()

    dataset = ImagesDataset(x, transforms=BASIC_TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    preds_collector = []

    model = resnet50_animal()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # run the forward step
            logits = model.forward(batch["image"].to(device))
            # apply softmax so that model outputs are in range [0,1]
            preds = torch.nn.functional.softmax(logits, dim=1)
            # store this batch's predictions in df
            preds_df = pd.DataFrame(
                preds.detach().cpu().numpy(),
                index=batch["image_id"],
                columns=[
                    'antelope_duiker',
                    'bird',
                    'blank',
                    'civet_genet',
                    'hog',
                    'leopard',
                    'monkey_prosimian',
                    'rodent'
                ],
            )
            preds_collector.append(preds_df)

    submission_df = pd.concat(preds_collector)
    submission_df.to_csv(prediction_path)


if __name__ == '__main__':
    fire.Fire(main)