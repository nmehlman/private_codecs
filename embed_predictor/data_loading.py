import os
import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class EmbeddingDataset(Dataset):
    def __init__(self, dataset: str = "expresso", codec: str = "encodec", split: str = "train", data_type: str = "quantized_embedding", data_root: str = None):
        
        # data_type can be "codes", "raw_embedding" or "quantized_embedding"

        self.data_type = data_type
        
        if data_root is None:
            data_root = f"/data/nmehlman/priv-codecs-data/emotion_embeddings/{dataset}/{codec}/{split}/"
        
        self.data_root = data_root
        self.all_files = sorted(os.listdir(data_root))
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        with open(os.path.join(self.data_root, self.all_files[index]), "rb") as f:
            sample = pickle.load(f)
            
            features = sample[self.data_type]
            
            label = sample["label"]

            length = features.shape[-1]
            
            return features, torch.tensor(label, dtype=torch.long), length
        
    @staticmethod
    def collate_function(batch):
        
        features = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch], dim=0)
        lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
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
        

        return batch_features, labels, lengths

if __name__ == "__main__":
    
    dataset = EmbeddingDataset(dataset="msp_podcast", codec="encodec", split="train", data_type="raw_embedding")
    print(f"Dataset size: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_function)
    
    for batch_features, batch_labels, batch_lengths in dataloader:
        print(f"Batch features shape: {batch_features.shape}, Batch labels shape: {batch_labels.shape}, Batch lengths: {batch_lengths}")
        print(batch_labels)
        break  # Just to print the first batch