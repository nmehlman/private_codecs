from disentangle.codec_data import EmbeddingDataset
import torch

VP_EMOTION_LABELS = [
        'Anger', # 0
        'Contempt', # 1
        'Disgust', # 2
        'Fear', # 3
        'Happiness', # 4 
        'Neutral', # 5
        'Sadness', # 6
        'Surprise', # 7
        'Other' # 8
    ]

if __name__ == "__main__":
    
    data_path = "/project2/shrikann_35/DATA/expresso/codec_feats/encodec"
    emotion_model = "wavlm"
    split = "train"
    save_path = f"../misc/emotion_prototypes_expresso_{split}_{emotion_model}.pt"
    
    dataset = EmbeddingDataset(
        dataset_path=data_path,
        split=split,
        emotion_model=emotion_model
    )
    
    grouped_embeddings = dict.fromkeys(VP_EMOTION_LABELS)
    
    for (_, emotion_emb, emotion_lab, _) in dataset: # Group embeddings by emotion label
        
        emotion_lab_str = VP_EMOTION_LABELS[emotion_lab]
        
        if grouped_embeddings[emotion_lab_str] is None:
            grouped_embeddings[emotion_lab_str] = []
        
        grouped_embeddings[emotion_lab_str].append(emotion_emb)
    
    # Compute mean prototype for each emotion
    emotion_prototypes = {
        emotion_lab: torch.mean(torch.stack(emb_list), dim=0) 
        for emotion_lab, emb_list in grouped_embeddings.items() if emb_list is not None
    }
        
    torch.save(emotion_prototypes, save_path)
        
        
    
    
    