from disentangle.lightning import DisentanglementAE
from network.models import VoxProfileEmotionModel
from data.expresso import ExpressoDataset
from data.msp_podcast import MSPPodcastDataset
import argparse

if __name__ == "__main__":
    
    ckpt_path = ""
    
    pl_model = DisentanglementAE.load_from_checkpoint(ckpt_path)
    
    # Setup dataset and dataloader
    