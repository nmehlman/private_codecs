import torch
import sys

sys.path.append("/home1/nmehlman/private_codecs/peft-ser/package")
import peft_ser

model = peft_ser.load_model("whisper-base-lora-16-conv", cache_folder="/project2/shrikann_35/nmehlman/models/")

data = torch.zeros([1, 2*16000])
output = model(data, return_features=False)
logits = torch.softmax(output, dim=1)
