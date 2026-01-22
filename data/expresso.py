import torch
from torch.utils.data import Dataset
import os
import re
import torchaudio

EXPRESSO_SR = 16000 # Assumes resampled audio, original is 48kHz

EXPRESSO_TO_VP_LABEL_MAPPING = {
    'angry': 0, #'Anger'
    'disgusted': 2, # 'Disgust'
    'fearful': 3, # 'Fear'
    'happy': 4, # 'Happiness'
    'sad': 6, # 'Sadness   
    'calm': 5, # 'Neutral'
    'sympathetic': 8, # 'Other'
    'default': 5, # 'Neutral'
    'laughing': 4, # 'Happiness'    
    'narration': 5, # 'Neutral'
}

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

LIMIT_EMOTIONS = list(EXPRESSO_TO_VP_LABEL_MAPPING.keys()) # Only include emotion labels that map to VoxProfile

class ExpressoDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        resample_rate: int = 16000,
        limit_emotions: list = LIMIT_EMOTIONS, # type: ignore
        label_mapping: dict = EXPRESSO_TO_VP_LABEL_MAPPING,
        target_length_s: int = 5,
        audio_dir: str = "audio_16khz",
        load_read_only: bool = False,
    ):

        self.data_dir = data_dir
        self.split = split
        self.resample_rate = resample_rate
        self.target_length_s = target_length_s
        self.audio_dir = audio_dir
        self.label_mapping = label_mapping

        splits_file_path = os.path.join(data_dir, "splits", f"{split}.txt")
        self.split_files = open(splits_file_path, "r").readlines()[1:]

        # Apply filtering based on emotion
        filtered_files = []
        for file in self.split_files:
            emotion = file.strip().split("\t")[0].split("_")[1]
            if emotion in limit_emotions:
                filtered_files.append(file)
        self.split_files = filtered_files

        # Parse VAD segments file into a dict: {name: [(start, end), ...]}
        vad_segments = {}
        vad_path = os.path.join(data_dir, "VAD_segments.txt")
        with open(vad_path, "r") as f:
            for line in f.readlines()[3:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                name = parts[0]
                segments = []
                matches = re.findall(r"\(([^,]+),\s*([^)]+)\)", line)
                for start_str, end_str in matches:
                    try:
                        start = float(start_str)
                        end = float(end_str)
                        segments.append((start, end))
                    except ValueError:
                        continue
                vad_segments[name] = segments
        self.vad_segments = vad_segments

        if resample_rate != EXPRESSO_SR:
            self.resample = torchaudio.transforms.Resample(EXPRESSO_SR, resample_rate)
        else:
            self.resample = None

        # Load transcripts
        transcript_path = os.path.join(data_dir, "read_transcriptions.txt")
        self.transcripts = {}
        with open(transcript_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    fname = parts[0]
                    transcript = parts[1]
                    self.transcripts[fname] = transcript

        # Build comprehensive sample index including VAD segments
        self.sample_index = []
        for file_info in self.split_files:
            fname = file_info.strip().split("\t")[0]
            spk, emotion, id = fname.split("_")[:3]

            if "-" in spk:  # Conversation emotion
                
                if load_read_only:
                    continue  # Skip conversational speech if loading read only

                # Add separate entries for each VAD segment in each channel
                for channel in [1, 2]:
                    vad_key = f"{fname}/channel{channel}"
                    if vad_key in self.vad_segments:
                        segments = self.vad_segments[vad_key]
                        for segment_idx, (start_time, end_time) in enumerate(segments):
                            self.sample_index.append(
                                {
                                    "file_info": file_info,
                                    "fname": fname,
                                    "spk": spk,
                                    "emotion": emotion,
                                    "id": id,
                                    "is_convo": True,
                                    "channel": channel,
                                    "vad_segment": (start_time, end_time),
                                    "segment_idx": segment_idx,
                                }
                            )
                    else:
                        # If no VAD data, add one entry per channel with full audio
                        self.sample_index.append(
                            {
                                "file_info": file_info,
                                "fname": fname,
                                "spk": spk,
                                "emotion": emotion,
                                "id": id,
                                "is_convo": True,
                                "channel": channel,
                                "vad_segment": None,
                                "segment_idx": None,
                            }
                        )
            else:  # Read emotion
                self.sample_index.append(
                    {   
                        "file_info": file_info,
                        "fname": fname,
                        "spk": spk,
                        "emotion": emotion,
                        "id": id,
                        "transcript": self.transcripts[fname],
                        "is_convo": False,
                        "channel": None,
                        "vad_segment": None,
                        "segment_idx": None,
                    }
                )

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):

        sample_info = self.sample_index[idx]
        fname = sample_info["fname"]
        spk = sample_info["spk"]
        emotion = sample_info["emotion"]
        id = sample_info["id"]
        is_convo = sample_info["is_convo"]

        if is_convo:
            audio_path = os.path.join(
                self.data_dir,
                self.audio_dir,
                "conversational",
                spk,
                emotion,
                f"{fname}.wav",
            )
        else:
            if 'longform' in fname:
                    audio_path = os.path.join(
                    self.data_dir,
                    self.audio_dir,
                    "read",
                    spk,
                    emotion,
                    "longform",
                    f"{fname}.wav",
                )
            else:
                audio_path = os.path.join(
                    self.data_dir,
                    self.audio_dir,
                    "read",
                    spk,
                    emotion,
                    "base",
                    f"{fname}.wav",
                )

        audio, sr = torchaudio.load(audio_path)
        assert sr == EXPRESSO_SR, f"Expected sample rate {EXPRESSO_SR}, but got {sr}"

        if self.resample is not None:
            audio = self.resample(audio)

        # Handle conversational speech with specific VAD segment
        if is_convo:
            channel = sample_info["channel"]
            vad_segment = sample_info["vad_segment"]

            if vad_segment is not None:
                start_time_seg, end_time = vad_segment

                # Convert time to sample indices
                start_sample = int(start_time_seg * self.resample_rate)
                end_sample = int(end_time * self.resample_rate)

                # Extract the segment from the specified channel
                start_sample = max(0, start_sample)
                end_sample = min(audio.size(1), end_sample)
                audio = audio[
                    channel - 1 : channel, start_sample:end_sample
                ]  # Select specific channel
                filename = f"{fname}_channel{channel}_seg{sample_info['segment_idx']}"
            else:
                # If no VAD segment, use the specified channel and full audio
                audio = audio[channel - 1 : channel, :]
                filename = f"{fname}_channel{channel}"


        # Handle read speech with existing start/end parsing (if present)
        elif not is_convo:
            finfo = sample_info["file_info"].strip().split("\t")
            if len(finfo) > 1:
                start_end = finfo[1].strip()
                if start_end and "," in start_end:
                    start_str, end_str = start_end.split(",")
                    try:
                        start = float(start_str.strip("()"))
                        end = float(end_str.strip("()"))

                        start_sample = int(start * self.resample_rate)
                        end_sample = int(end * self.resample_rate)

                        start_sample = max(0, start_sample)
                        end_sample = min(audio.size(1), end_sample)
                        audio = audio[:, start_sample:end_sample]
                    except ValueError:
                        pass  # Use full audio if parsing fails
                    
            filename = fname  # Use filename without extension

        # Pad or truncate audio to target length (always take first target_length_s seconds)
        target_length_samples = int(self.target_length_s * self.resample_rate)
        audio_length = audio.size(1)

        if audio_length < target_length_samples:
            padding = target_length_samples - audio_length
            audio = torch.nn.functional.pad(audio, (0, padding))
            length = audio_length

        elif audio_length >= target_length_samples:
            # Take the first target_length_s seconds instead of random cropping
            audio = audio[:, :target_length_samples]
            length = target_length_samples

        return {
            "audio": audio,
            "speaker": spk,
            "emotion": self.label_mapping[emotion],
            "id": id,
            "length": length,
            "filename": filename,
            "transcript": sample_info.get("transcript", ""),
        }

    @staticmethod
    def collate_function(batch):
        max_len = max(item["length"] for item in batch)
        padded = []
        lengths = []
        for item in batch:
            a = item["audio"]
            if a.size(1) < max_len:
                pad = max_len - a.size(1)
                a = torch.nn.functional.pad(a, (0, pad))
            padded.append(a)
            lengths.append(item["length"])

        audio = torch.concat(padded, dim=0)  # (batch, channels, max_len)
        emotions = torch.tensor([item["emotion"] for item in batch], dtype=torch.long)
        speakers = [item["speaker"] for item in batch]
        ids = [item["id"] for item in batch]
        filenames = [item["filename"] for item in batch]
        lengths = torch.tensor(lengths, dtype=torch.long)
        transcripts = [item["transcript"] for item in batch]

        return {
            "audio": audio,
            "speaker": speakers,
            "emotion": emotions,
            "id": ids,
            "length": lengths,
            "filenames": filenames,
            "transcript": transcripts,
        }

if __name__ == "__main__":

    import tqdm

    # Example usage
    dataset = ExpressoDataset(
        data_dir="/data1/open_data/expresso/", split="train", resample_rate=16000, load_read_only=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=ExpressoDataset.collate_function
    )

    for batch in tqdm.tqdm(data_loader, total=len(data_loader)):
        print(batch["audio"].shape)
        print(batch["emotion"])
        print(batch["transcript"])