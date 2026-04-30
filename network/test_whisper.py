from src.model.emotion.whisper_emotion import WhisperWrapper
import torch

if __name__ == "__main__":

    x = torch.randn(2, 16000 * 3)
    length = torch.tensor([16000 * 1, 16000 * 2])

    model = WhisperWrapper()

    print("With lengths:")
    _, batch_emb, _, _, _, _ = model(x, length=length)

    singleton_embs = []
    for i in range(x.shape[0]):
        _, emb, _, _, _, _ = model(x[i:i+1], length=length[i:i+1])
        singleton_embs.append(emb)

    singleton_embs = torch.cat(singleton_embs, dim=0)
    print(batch_emb == singleton_embs)  # Should be True if they are the same
    print(batch_emb[0], singleton_embs[0])  # Print to visually confirm they are the same

    print("Without lengths:")
    _, batch_emb, _, _, _, _ = model(x, length=None)  

    singleton_embs = []
    for i in range(x.shape[0]):
        _, emb, _, _, _, _ = model(x[i:i+1], length=None)
        singleton_embs.append(emb)  
    singleton_embs = torch.cat(singleton_embs, dim=0)
    print(batch_emb == singleton_embs)  # Should be True if they are the same
    print(batch_emb[0], singleton_embs[0])  # Print to visually confirm
