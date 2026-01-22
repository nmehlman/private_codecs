from network.models import VoxProfileEmotionModel, VOX_PROFILE_SR
from data.expresso import ExpressoDataset, EXPRESSO_TO_VP_LABEL_MAPPING, EXPRESSO_SR
from data.msp_podcast import MSPPodcastDataset, MSP_SR
from network.codec import ENCODEC_SR, HIFICODEC_SR, BIGCODEC_SR, EnCodec, HifiCodec, BigCodec
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from embed_predictor.embed_classifier import EmbeddingClassifier
import torch
import os
import tqdm
import torchaudio

CODECS = {
    "encodec": (EnCodec, ENCODEC_SR),
    "hificodec": (HifiCodec, HIFICODEC_SR),
    "bigcodec": (BigCodec, BIGCODEC_SR),
}

DATASETS = {
    "expresso": (ExpressoDataset, EXPRESSO_SR),
    "msp_podcast": (MSPPodcastDataset, MSP_SR),
}

class MaskedCategoricalEmotionInference:

    def __init__(self, 
                 embed_model_checkpoint: str,
                 latent_dim: int = 128,
                 sigma: float = 1.0,
                 codec: str = "encodec", 
                 codec_args: dict = {},
                 dataset: str = "expresso", 
                 data_args: dict = {},
                 device: str = "cpu",
                 audio_save_path: str = None,
                 audio_batches_to_save: int = 0
                 ) -> None:

        # Load codec
        codec_class, codec_sr = CODECS[codec]
        self.codec = codec_class(**codec_args, device=device)

        # Load emotion model
        self.emotion_model = VoxProfileEmotionModel(device=device)

        # Load embedding classifier
        self.embed_classifier = EmbeddingClassifier.load_from_checkpoint(embed_model_checkpoint, input_dim=latent_dim, num_classes=9).to(device)

        # Load ASR model
        self.asr_processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        self.asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large").to(device)
        self.asr_model.config.forced_decoder_ids = None
        
        dataset_class, dataset_sr = DATASETS[dataset]
        self.dataset = dataset_class(**data_args)
        assert self.dataset.resample_rate == dataset_sr, "Dataset sample rate mismatch."
        
        self.sigma = sigma
        self.audio_save_path = audio_save_path
        self.audio_batches_to_save = audio_batches_to_save
        self.device = device
        self.codec_sr = codec_sr
        self.dataset_sr = dataset_sr
        self.vox_profile_sr = VOX_PROFILE_SR
    
    def _get_asr_transcriptions(self, audio: torch.Tensor) -> list:
        transcripts = []
        for i in range(audio.shape[0]):
            input_features = self.asr_processor(audio[i].cpu(), sampling_rate=self.dataset_sr, return_tensors="pt").input_features 
            predicted_ids = self.asr_model.generate(input_features.to(self.device)).cpu()
            transcription = self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcripts.append(transcription)

        return transcripts

    def _process_batch(self, batch: dict) -> dict:
        
        raw_audio = batch["audio"].to(self.device) 
        emotion_labels = batch["emotion"].to(self.device)
        lengths = batch["length"].to(self.device)

        transcripts = batch["transcript"]
        assert transcripts[0] != "", "Empty transcript found in batch."
        
        # Get predicted and ASR for raw audio
        with torch.no_grad():
            emotion_preds_raw = self.emotion_model(raw_audio.clone(), sr=self.dataset_sr, lengths=lengths)
            transcriptions_raw = self._get_asr_transcriptions(raw_audio.clone())
        
        # Run noisy reconstuction
        embeddings = self.codec.encode(raw_audio, sr=self.dataset_sr)
        _, quantized_embeddings = self.codec.quantize(embeddings)
        # TODO: explore adding noise in the embedding space before quantization

        noisy_embedding = []
        for q, l in zip(quantized_embeddings, lengths):
            q = q.clone().detach()
            J = torch.autograd.functional.jacobian(lambda x: self.embed_classifier(x, l.unsqueeze(0)).norm(), q.unsqueeze(0)).squeeze() # Jacobian of embed_classifier norm wrt quantized embeddings
            J = J.abs() / J.abs().max() # Normalize
            assert torch.isnan(J).sum() == 0, "NaN values found in Jacobian."
            noise = torch.randn_like(q) * self.sigma * J # Add shaped noise
            q += noise
            noisy_embedding.append(q)
            
        noisy_embedding = torch.stack(noisy_embedding, dim=0)
        quantized_noisy, _ = self.codec.quantize(noisy_embedding)
        recon_audio = self.codec.decode(quantized_noisy)
        
        # Resample back to dataset rate for length consistency
        if self.codec_sr != self.dataset_sr:
            recon_audio = torchaudio.functional.resample(
                recon_audio, orig_freq=self.codec_sr, new_freq=self.dataset_sr
            )
            
        with torch.no_grad():
            emotion_preds_recon = self.emotion_model(recon_audio.clone(), sr=self.dataset_sr, lengths=lengths)
            transcriptions_recon = self._get_asr_transcriptions(recon_audio.clone())

        return {
            "recon_audio": recon_audio,
            "emotion_preds_raw": emotion_preds_raw,
            "emotion_preds_recon": emotion_preds_recon,
            "true_labels": emotion_labels,
            "transcriptions_raw": transcriptions_raw,
            "true_transcripts": transcripts,
            "transcriptions_recon": transcriptions_recon,
            "file_names": batch["filenames"]
        }
    
    def run(self, batch_size: int = 8) -> dict:

        if self.audio_save_path is not None:
            os.makedirs(self.audio_save_path, exist_ok=True)
            assert self.audio_batches_to_save > 0, "audio_batches_to_save must be > 0 if audio_save_path is specified."
        
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, collate_fn=self.dataset.collate_function)
        
        all_emotion_preds_raw = []
        all_emotion_preds_recon = []
        all_true_emotion_labels = []
        
        all_transcriptions_raw = []
        all_true_transcripts = []
        all_transcriptions_recon = []

        all_file_names = []
        
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Running Inference"):
            
            results = self._process_batch(batch)
            
            all_emotion_preds_raw.append(results["emotion_preds_raw"].cpu())
            all_emotion_preds_recon.append(results["emotion_preds_recon"].cpu())
            all_true_emotion_labels.append(results["true_labels"].cpu())
            
            all_transcriptions_raw.extend(results["transcriptions_raw"])
            all_transcriptions_recon.extend(results["transcriptions_recon"])
            all_true_transcripts.extend(results["true_transcripts"])
            
            all_file_names.extend(results["file_names"])
            
            if self.audio_save_path is not None and i < self.audio_batches_to_save:
                for j, audio in enumerate(results["recon_audio"]):
                    filename = results["file_names"][j]
                    save_path = os.path.join(self.audio_save_path, f"{filename}.wav")
                    print(f"Saving reconstructed audio to {save_path}")
                    torchaudio.save(save_path, audio.cpu().unsqueeze(0), sample_rate=self.dataset_sr)
        
        return {
            "emotion_preds_raw": torch.cat(all_emotion_preds_raw, dim=0),
            "emotion_preds_recon": torch.cat(all_emotion_preds_recon, dim=0),
            "true_emotion_labels": torch.cat(all_true_emotion_labels, dim=0),
            "transcriptions_raw": all_transcriptions_raw,
            "transcriptions_recon": all_transcriptions_recon,
            "true_transcripts": all_true_transcripts,
            "file_names": all_file_names
        }

class CategoricalEmotionInference:

    def __init__(self, 
                 codec: str = "encodec", 
                 codec_args: dict = {},
                 dataset: str = "expresso", 
                 data_args: dict = {},
                 device: str = "cpu") -> None:

        codec_class, codec_sr = CODECS[codec]
        self.codec = codec_class(**codec_args, device=device)
        
        dataset_class, dataset_sr = DATASETS[dataset]
        self.dataset = dataset_class(**data_args)
        assert self.dataset.resample_rate == dataset_sr, "Dataset sample rate mismatch."
        
        self.emotion_model = VoxProfileEmotionModel(device=device)
        
        self.device = device
        self.codec_sr = codec_sr
        self.dataset_sr = dataset_sr
        self.vox_profile_sr = VOX_PROFILE_SR
        
    def _process_batch(self, batch: dict) -> dict:
        
        raw_audio = batch["audio"].to(self.device) 
        emotion_labels = batch["emotion"].to(self.device)
        lengths = batch["length"].to(self.device)
        
        # Get predicted emotion for raw audio
        with torch.no_grad():
            emotion_preds_raw = self.emotion_model(raw_audio.clone(), sr=self.dataset_sr, lengths=lengths)
            
        # Get predicted emotion for codec reconstructed audio
        with torch.no_grad():
            codec_outputs = self.codec(raw_audio, sr=self.dataset_sr)
            recon_audio = codec_outputs["recon_audio"]

            # Resample back to dataset rate for length consistency
            if self.codec_sr != self.dataset_sr:
                recon_audio = torchaudio.functional.resample(
                    recon_audio, orig_freq=self.codec_sr, new_freq=self.dataset_sr
                )
            
            # Get predictions
            preds_recon = self.emotion_model(recon_audio.clone(), sr=self.dataset_sr, lengths=lengths)

        return {
            "emotion_preds_raw": emotion_preds_raw,
            "emotion_preds_recon": preds_recon,
            "true_emotion_labels": emotion_labels,
            "file_names": batch["filenames"]
        }
    
    def run(self, batch_size: int = 8) -> dict:
        
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, collate_fn=self.dataset.collate_function)
        
        all_emotion_preds_raw = []
        all_emotion_preds_recon = []
        all_true_emotion_labels = []
        all_file_names = []
        
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Running Inference"):
            results = self._process_batch(batch)
            all_emotion_preds_raw.append(results["emotion_preds_raw"].cpu())
            all_emotion_preds_recon.append(results["emotion_preds_recon"].cpu())
            all_true_emotion_labels.append(results["true_emotion_labels"].cpu())
            all_file_names.extend(results["file_names"])
        
        return {
            "emotion_preds_raw": torch.cat(all_emotion_preds_raw, dim=0),
            "emotion_preds_recon": torch.cat(all_emotion_preds_recon, dim=0),
            "true_emotion_labels": torch.cat(all_true_emotion_labels, dim=0),
            "file_names": all_file_names
        }
            
if __name__ == "__main__":
    
    inference = CategoricalEmotionInference(
        codec="encodec",
        dataset="expresso",
        data_args={"data_dir": "/data1/open_data/expresso/", "split": "test"},
        device="cuda:0"
    )
    
    results = inference.run(batch_size=16)
    
    import pickle
    with open("/home/nmehlman/private-codecs/test_outputs/emotion_inference_results.pkl", "wb") as f:
        pickle.dump(results, f)