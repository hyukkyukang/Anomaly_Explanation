from torch.utils.data import DataLoader

from DBAnomTransformer.data_factory.dataset.dbsherlock import DBSherlockDataset
from DBAnomTransformer.data_factory.dataset.eda import EDADataset
from DBAnomTransformer.data_factory.dataset.msl import MSLDataset
from DBAnomTransformer.data_factory.dataset.psm import PSMDataset
from DBAnomTransformer.data_factory.dataset.smap import SMAPDataset
from DBAnomTransformer.data_factory.dataset.smd import SMDDataset


def get_dataloader(
    dataset_path: str,
    batch_size: int,
    win_size: int = 25,
    step: int = 25,
    mode: str = "test",
    dataset: str = "EDA",
) -> DataLoader:
    if dataset == "SMD":
        dataset = SMDDataset(
            dataset_path=dataset_path, win_size=win_size, step=step, mode=mode
        )
    elif dataset == "MSL":
        dataset = MSLDataset(
            dataset_path=dataset_path, win_size=win_size, step=1, mode=mode
        )
    elif dataset == "SMAP":
        dataset = SMAPDataset(
            dataset_path=dataset_path, win_size=win_size, step=1, mode=mode
        )
    elif dataset == "PSM":
        dataset = PSMDataset(
            dataset_path=dataset_path, win_size=win_size, step=1, mode=mode
        )
    elif dataset == "DBS":
        dataset = DBSherlockDataset(
            dataset_path=dataset_path, win_size=win_size, step=step, mode=mode
        )
    elif dataset == "EDA":
        dataset = EDADataset(
            dataset_path=dataset_path, win_size=win_size, step=step, mode=mode
        )

    do_shuffle = mode == "train"

    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=do_shuffle, num_workers=0
    )
    return data_loader
