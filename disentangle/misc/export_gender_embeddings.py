"""Wrapper for the gender classifier from https://github.com/JaesungHuh/voice-gender-classifier"""

# Need to add to path
import sys
sys.path.append("/home1/nmehlman/PESSA_Project/repos")
from voice_gender_classifier.model import ECAPA_gender # type: ignore

import torch
from typing import Tuple, Union
from fileinput import filename
from data.expresso import ExpressoDataset, EXPRESSO_SR
from data.msp_podcast import MSPPodcastDataset, MSP_SR
from data.vox1 import Vox1Dataset, VOX1_SR
from network.codec import HifiCodec, EnCodec, BigCodec, HIFICODEC_SR, ENCODEC_SR, BIGCODEC_SR
import tqdm
import torch
import argparse
import pickle
import os
import yaml

from network.models import VoxProfileAgeSexModel

class VoiceGenderClassifier(torch.nn.Module):

    sample_rate = 16000

    def __init__(self, device: Union[torch.device, str] = torch.device("cpu")) -> None:

        self.device = device
        self.model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
        self.model.eval().to(self.device)

    def forward(self, x: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, dict]:

        x = self.model.logtorchfbank(x)

        x = self.model.conv1(x)
        x = self.model.relu(x)
        x = self.model.bn1(x)

        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x+x1)
        x3 = self.model.layer3(x+x1+x2)

        x = self.model.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.model.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.model.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.model.bn5(x)
        x = self.model.fc6(x)
        x = self.model.bn6(x)
        z = self.model.relu(x)
        y = self.model.fc7(z)

        return z, {"gender_logits": y}
    
    @property
    def embedding_dim(self) -> int:
        return 192 

DATASETS = {
    "expresso": (ExpressoDataset, EXPRESSO_SR),
    "msp_podcast": (MSPPodcastDataset, MSP_SR),
    "vox1": (Vox1Dataset, VOX1_SR),
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Export codec embeddings.")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    dataset_name = config["dataset_name"]
    codec_name = config["codec_name"]

    print(f"Exporting:\n\tGender\n\tDataset: {dataset_name}\n\tDevice: {config['device']}")
    
    save_root = config["save_path"]
    os.makedirs(save_root, exist_ok=True)

    dataset_class, dataset_sr = DATASETS[dataset_name]

    sex_model = VoiceGenderClassifier(device=config["device"])

    dataset = dataset_class(**config["dataset"])

    for sample in tqdm.tqdm(dataset, total=len(dataset), desc="Exporting Data"):
        
        audio = sample["audio"].to(config["device"])
        filename = sample["filename"]
        length = sample["length"]
        speaker = sample["speaker"]
        session = sample.get("session", "unknown_session")  # Vox1 has session info, others may not

        sex_embedding, _ = sex_model(
            audio, lengths=torch.tensor([length]).to(config["device"])
        )
        
        save_path = os.path.join(save_root, f"{speaker}_{session}_{filename}.pkl")
        assert os.path.exists(save_path)

