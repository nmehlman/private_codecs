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

    # Compute intra-class similarity (within each emotion)
    intra_similarity = {}
    for emotion_lab, emb_list in grouped_embeddings.items():
        if emb_list is not None:
            emb_stack = torch.stack(emb_list)
            # Normalize for cosine similarity
            emb_norm = torch.nn.functional.normalize(emb_stack, p=2, dim=1)
            cosine_sim = torch.mm(emb_norm, emb_norm.t())
            # Remove diagonal and compute mean
            mask = ~torch.eye(cosine_sim.shape[0], dtype=torch.bool)
            intra_similarity[emotion_lab] = {
                'cosine': cosine_sim[mask].mean().item(),
                'l2': torch.cdist(emb_stack, emb_stack).mean().item()
            }

    # Compute inter-class similarity (between different emotions)
    inter_similarity = {}
    emotion_labs = list(grouped_embeddings.keys())
    for i, lab1 in enumerate(emotion_labs):
        for lab2 in emotion_labs[i+1:]:
            if grouped_embeddings[lab1] is not None and grouped_embeddings[lab2] is not None:
                emb_stack1 = torch.stack(grouped_embeddings[lab1])
                emb_stack2 = torch.stack(grouped_embeddings[lab2])
                emb_norm1 = torch.nn.functional.normalize(emb_stack1, p=2, dim=1)
                emb_norm2 = torch.nn.functional.normalize(emb_stack2, p=2, dim=1)
                cosine_sim = torch.mm(emb_norm1, emb_norm2.t()).mean().item()
                l2_dist = torch.cdist(emb_stack1, emb_stack2).mean().item()
                inter_similarity[f"{lab1}-{lab2}"] = {
                    'cosine': cosine_sim,
                    'l2': l2_dist
                }

    print("Intra-class similarity:", intra_similarity)
    print("Inter-class similarity:", inter_similarity)
        
    torch.save(emotion_prototypes, save_path)
        
        
    
    
    