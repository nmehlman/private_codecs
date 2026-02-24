from disentangle.codec_data import EmbeddingDataset
import torch
import json

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
    
    data_path = "/project2/shrikann_35/DATA/expresso/codec_feats/"
    emotion_model = "wavlm"
    codec = "encodec"
    split = "train"
    
    mode = "average"  # "average" or "random"
    min_dist_random = 0.1  # Minimum cosine distance between randomly selected samples for "random" mode
    max_random_samples = 5000  # Maximum number of random samples to try before giving up

    save_path = f"./emotion_prototypes_expresso_{split}_{emotion_model}_{mode}.pt"
    save_stats = False
    
    dataset = EmbeddingDataset(
        dataset_path=data_path,
        split=split,
        codec=codec,
        emotion_model=emotion_model
    )
    
    grouped_embeddings = dict.fromkeys(VP_EMOTION_LABELS)
    
    for (_, emotion_emb, emotion_lab, _) in dataset: # Group embeddings by emotion label
        
        emotion_lab_str = VP_EMOTION_LABELS[emotion_lab]
        
        if grouped_embeddings[emotion_lab_str] is None:
            grouped_embeddings[emotion_lab_str] = []
        
        grouped_embeddings[emotion_lab_str].append(emotion_emb)
    
    # Compute mean prototype for each emotion
    if mode == "average":
        emotion_prototypes = {
            emotion_lab: torch.mean(torch.stack(emb_list), dim=0) 
            for emotion_lab, emb_list in grouped_embeddings.items() if emb_list is not None
        }
    elif mode == "random":
        
        distance_criterion_met = False
        samples_taken = 0

        while not distance_criterion_met:
            samples_taken += 1
            
            emotion_prototypes = {}
            for emotion_lab, emb_list in grouped_embeddings.items():
                if emb_list is not None:
                    emotion_prototypes[emotion_lab] = emb_list[torch.randint(len(emb_list), (1,)).item()]
            
            # Check distance criterion
            all_embeddings = list(emotion_prototypes.values())
            all_embeddings_norm = [torch.nn.functional.normalize(emb, p=2, dim=0) for emb in all_embeddings]
            cosine_sim_matrix = torch.mm(torch.stack(all_embeddings_norm), torch.stack(all_embeddings_norm).t())
            cosine_dist_matrix = 1 - cosine_sim_matrix
            upper_triangular_indices = torch.triu_indices(cosine_dist_matrix.size(0), cosine_dist_matrix.size(1), offset=1)
            distances = cosine_dist_matrix[upper_triangular_indices[0], upper_triangular_indices[1]]
            if (distances >= min_dist_random).all():
                distance_criterion_met = True
                print(f"Found prototypes meeting distance criterion after {samples_taken} random samples.")

            elif samples_taken >= max_random_samples:
                raise ValueError(f"Could not find random prototypes meeting distance criterion after {max_random_samples} samples. Consider lowering min_dist_random or increasing max_random_samples.")

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

    # Compute inter-prototype similarity (between different emotion prototypes)
    inter_prototype_similarity = {}
    prototype_labs = list(emotion_prototypes.keys())
    for i, lab1 in enumerate(prototype_labs):
        for lab2 in prototype_labs[i+1:]:
            emb1 = emotion_prototypes[lab1]
            emb2 = emotion_prototypes[lab2]
            emb_norm1 = torch.nn.functional.normalize(emb1, p=2, dim=0)
            emb_norm2 = torch.nn.functional.normalize(emb2, p=2, dim=0)
            cosine_sim = torch.dot(emb_norm1, emb_norm2).item()
            l2_dist = torch.norm(emb1 - emb2).item()
            inter_prototype_similarity[f"{lab1}-{lab2}"] = {
                'cosine': cosine_sim,
                'l2': l2_dist
            }

    stats = {
        'intra_similarity': intra_similarity,
        'inter_similarity': inter_similarity,
        'prototype_similarity': inter_prototype_similarity
    }
    
    if save_stats:
        stats_path = save_path.replace('.pt', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
    torch.save(emotion_prototypes, save_path)
        
        
    
    
    
