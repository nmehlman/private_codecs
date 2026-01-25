import os
import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from typing import Dict, Callable, Union

class EmbeddingDataset(Dataset):
    def __init__(
            self, 
            dataset_path: str, 
            split: str = "train", 
            input_type: str = "quantized_embedding", # input_type can be "codes", "raw_embedding" or "quantized_embedding"
            emotion_model: str = "wavlm"):

        self.input_type = input_type
        self.emotion_model = emotion_model
        
        data_root = os.path.join(dataset_path, split)
        
        self.data_root = data_root
        self.all_files = sorted(os.listdir(data_root))
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        
        with open(os.path.join(self.data_root, self.all_files[index]), "rb") as f:
            
            sample = pickle.load(f)
            
            features = sample[self.input_type]
            emotion_lab = sample[f"{self.emotion_model}_emotion_logits"].argmax().item()
            emotion_emb = sample[f"{self.emotion_model}_emotion_embedding"].detach()
            
            # Check for NaN values
            if torch.isnan(features).any():
                raise ValueError(f"NaN found in features at index {index}")
            if torch.isnan(emotion_emb).any():
                raise ValueError(f"NaN found in emotion_emb at index {index}")
            
            length = features.shape[-1]
            
            return (features, emotion_emb, emotion_lab, length)
        
    @staticmethod
    def collate_function(batch):
        
        features = [item[0] for item in batch]
        emotion_embs = torch.stack([item[1] for item in batch], dim=0)
        emotion_labs = torch.tensor([item[2] for item in batch], dtype=torch.long)
        lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)
        max_len = max(feats.shape[-1] for feats in features)
        
        padded_features = []
        for feats in features:
            length = feats.shape[-1]
            pad_size = max_len - length
            if pad_size > 0:
                pad_tensor = torch.zeros((*feats.shape[:-1], pad_size), dtype=feats.dtype)
                padded_feats = torch.cat([feats, pad_tensor], dim=-1)
            else:
                padded_feats = feats
            
            padded_features.append(padded_feats)
        
        batch_features = torch.stack(padded_features, dim=0)
        

        return batch_features, emotion_embs, emotion_labs, lengths
    

def get_dataloaders(
                    dataset_kwargs: Dict = {},
                    batch_size: int = 16,
                    train_frac: float = 1.0,
                    **dataloader_kwargs
                    ) -> Union[ DataLoader, Dict ]:

    """Generate dataloader(s) with option to split into train/val

    Args:
        DatasetClass (Dataset): dataset to use for generating loader
        dataset_kwargs (Dict): kwargs for dataset construction
        batch_size (int): batch size
        collate_fn (Union[Callable, None], optional): Function to use for batch collation. Defaults to None.
        train_frac (float, optional): fraction of data to use for train split. Defaults to 1.0
        dataloader_kwargs (Dict, optional): additional kwargs to pass to dataloader constructor

    Returns:
        loader(s) (Union[ DataLoader, Dict ]): single dataloader if train_frac = 1.0, or dict with train/val loaders
        if train_frac < 1.0
    """

    dset = EmbeddingDataset(**dataset_kwargs)

    if train_frac < 1.0:

        dset_size = len(dset)
        train_size = int(dset_size * train_frac)
        val_size = dset_size - train_size

        train_dset, val_dset = random_split(dset, [train_size, val_size])

        train_loader = DataLoader(
                                dataset = train_dset,
                                batch_size = batch_size,
                                collate_fn = EmbeddingDataset.collate_function,
                                **dataloader_kwargs
                            )
        
        val_loader = DataLoader(
                                dataset = val_dset,
                                batch_size = batch_size,
                                collate_fn = EmbeddingDataset.collate_function,
                                **dataloader_kwargs
                            )
        
        loaders = {"train": train_loader, "val": val_loader}

        return loaders

    else:

        loader = DataLoader(
                            dataset = dset,
                            batch_size = batch_size,
                            collate_fn = EmbeddingDataset.collate_function,
                            **dataloader_kwargs
                            )

        return loader


                    
    
if __name__ == "__main__":

    data_path = "/project2/shrikann_35/DATA/expresso/codec_feats/encodec"

    dataset = EmbeddingDataset(
        dataset_path=data_path,
        split="train",
        input_type="quantized_embedding",
        emotion_model="wavlm"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=EmbeddingDataset.collate_function
    )

    for batch in dataloader:
        features, emotion_embs, emotion_labs, lengths = batch
        print("Features shape:", features.shape)
        print("Emotion embeddings shape:", emotion_embs.shape)
        print("Emotion labels shape:", emotion_labs.shape)
        print("Lengths:", lengths)
        break