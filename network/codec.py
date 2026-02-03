from transformers import EncodecModel, AutoProcessor
import torch
import torchaudio
from academicodec.models.hificodec.vqvae_tester import VqvaeTester
from vq.codec_encoder import CodecEncoder
from vq.codec_decoder import CodecDecoder

ENCODEC_SR = 24000
HIFICODEC_SR = 16000
BIGCODEC_SR = 16000

class EnCodec:
    def __init__(self, device: str = "cpu", *args, **kwargs):
        
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.sample_rate = 24000
        self.device = device
        self.model.to(device) # type: ignore

    def __call__(self, audio: torch.Tensor, sr: int):
        
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)
        
        audio = audio.cpu().numpy()
        
        if audio.shape[0] >= 1:
            audio = [audio[i].squeeze() for i in range(audio.shape[0])]
            
        inputs = self.processor(raw_audio=audio, sampling_rate=self.sample_rate, return_tensors="pt")
        output = self.model(inputs["input_values"].to(self.device))
        recon_audio = output.audio_values
        audio_codes = output.audio_codes
        
        return {"recon_audio": recon_audio.squeeze(), "audio_codes": audio_codes.squeeze()}
    
    def encode(self, audio: torch.Tensor, sr: int):
        
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)
        
        audio = audio.cpu().numpy()
        
        if audio.shape[0] >= 1:
            audio = [audio[i].squeeze() for i in range(audio.shape[0])]
            
        inputs = self.processor(raw_audio=audio, sampling_rate=self.sample_rate, return_tensors="pt")

        with torch.no_grad():
            _, audio_scales, embeds = self.model.encode(inputs["input_values"].to(self.device), return_dict=False) # (1, B, K, T)

        assert isinstance(audio_scales, list)
        assert len(audio_scales) == 1
        assert audio_scales[0] is None

        return embeds.squeeze(0)
    
    def quantize(self, embeds: torch.Tensor):
        
        bandwidth = self.model.config.target_bandwidths[0]
        codes = self.model.quantizer.encode(embeds, bandwidth=bandwidth) 
        quantized_embeddings = self.model.quantizer.decode(codes)
        return codes.transpose(0, 1), quantized_embeddings # (B, K, T)
    
    def decode(self, audio_codes: torch.Tensor):
                
        with torch.no_grad():
            recon_audio = self.model.decode(audio_codes.unsqueeze(0), audio_scales=[None])[0]
        
        return recon_audio.squeeze()

class HifiCodec:
    def __init__(self, 
                 config_path: str = "/home1/nmehlman/private_codecs/AcademiCodec/egs/HiFi-Codec-16k-320d//config_16k_320d.json", 
                 model_path: str = "/project2/shrikann_35/nmehlman/models/HiFi-Codec-16k-320d", 
                 device: str = "cpu", 
                 *args, 
                 **kwargs
            ):
        
        self.model = VqvaeTester(config_path=config_path, model_path=model_path, sample_rate=HIFICODEC_SR).vqvae
        self.sample_rate = HIFICODEC_SR
        self.device = device
        self.model.to(device)
        
    
    def __call__(self, audio: torch.Tensor, sr: int):
        
        audio = audio.to(self.device)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)

        audio_codes = self.model.encode(audio).squeeze(1)
        recon_audio = self.model(audio_codes).squeeze(1)
        
        return {"recon_audio": recon_audio.squeeze(), "audio_codes": audio_codes.squeeze()}
    
    def encode(self, audio: torch.Tensor, sr: int):
        
        audio = audio.to(self.device)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)

        embeds = self.model.encoder(audio.unsqueeze(1)).squeeze(1)
        
        return embeds
    
    def quantize(self, embeds: torch.Tensor):

        batch_size = embeds.size(0)
        embeds = embeds.to(self.device)
        
        quantized_embeddings, _, c = self.model.quantizer(embeds)
        c = [code.reshape(batch_size, -1) for code in c]
        audio_codes = torch.stack(c, -1).transpose(1,2)

        return audio_codes, quantized_embeddings
    
    def decode(self, audio_codes: torch.Tensor):
        audio_codes = audio_codes.transpose(1,2)
        audio_codes = audio_codes.to(self.device)
        recon_audio = self.model(audio_codes).squeeze(1)
        
        return recon_audio

class BigCodec:
    def __init__(self, 
                 model_path: str = "/data1/nmehlman/models/BigCodec/bigcodec.pt", 
                 device: str = "cpu", 
                 *args, 
                 **kwargs
            ):
        
        ckpt = torch.load(model_path, map_location='cpu')
        
        self.encoder = CodecEncoder()
        self.encoder.load_state_dict(ckpt['CodecEnc'])
        self.encoder = self.encoder.eval().to(device)
        
        self.decoder = CodecDecoder()
        self.decoder.load_state_dict(ckpt['generator'])
        self.decoder = self.decoder.eval().to(device)

        self.device = device
        self.sample_rate = BIGCODEC_SR
        
    
    def __call__(self, audio: torch.Tensor, sr: int):
        
        audio = audio.to(self.device)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)

        audio = audio.unsqueeze(1)  # Add channel dimension
        embeds = self.encoder(audio)
        vq_post_emb, audio_codes, _ = self.decoder(embeds, vq=True)
        recon_audio = self.decoder(vq_post_emb, vq=False).squeeze().detach().cpu()
            
        return {"recon_audio": recon_audio.squeeze(), "audio_codes": audio_codes.squeeze()}

    def encode(self, audio: torch.Tensor, sr: int):
        
        audio = audio.to(self.device)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)

        audio = audio.unsqueeze(1)  # Add channel dimension
        embeds = self.encoder(audio)
        return embeds

    def quantize(self, embeds: torch.Tensor):
        quantized_embeddings, audio_codes, _ = self.decoder(embeds, vq=True)
        return audio_codes, quantized_embeddings

    def decode(self, audio_codes: torch.Tensor):
        quantized_embeddings = self.decoder.vq2emb(audio_codes.transpose(0,2)).permute(1,2,0)
        recon_audio = self.decoder(quantized_embeddings, vq=False).squeeze().detach().cpu()
        return recon_audio  


if __name__ == "__main__":
    
    from data.expresso import ExpressoDataset, EXPRESSO_SR
    import torchaudio
    import numpy as np
    
    dataset = ExpressoDataset(data_dir="/project2/shrikann_35/DATA/expresso/")

    samples = torch.stack([dataset[i]["audio"].squeeze() for i in np.random.randint(0, len(dataset), size=3)], dim=0)[:,:]  # 5 seconds
    print(f"Original samples shape: {samples.shape}")
    codec = EnCodec()  
    
    #output = codec(samples, sr=EXPRESSO_SR)
    #recon_audio = output["recon_audio"]

    embeds = codec.encode(samples, sr=EXPRESSO_SR)

    audio_codes, quantized_embeddings = codec.quantize(embeds)
    quantized_embeddings += torch.randn_like(quantized_embeddings) * 0.0  # Add noise if desired
    print(f"Quantized embeddings shape: {quantized_embeddings.shape}, Audio codes shape: {audio_codes.shape}")

    recon_audio = codec.decode(audio_codes)

    #for i in range(samples.shape[0]):
    #    torchaudio.save(f"/home/nmehlman/private-codecs/test_audio/original_{i}.wav", samples[i].unsqueeze(0), sample_rate=EXPRESSO_SR)
    #    torchaudio.save(f"/home/nmehlman/private-codecs/test_audio/reconstructed_{i}.wav", recon_audio[i].unsqueeze(0), sample_rate=codec.sample_rate)