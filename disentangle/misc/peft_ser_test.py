import torch
import sys

sys.path.append("/Users/nick/Desktop/Private Codecs/peft-ser/package")
import peft_ser

model = peft_ser.load_model("whisper-base-lora-16-conv", cache_folder="/project2/shrikann_35/nmehlman/models/")

#data = torch.zeros([1, 16000])
#output = model(data)
#logits = torch.softmax(output, dim=1)