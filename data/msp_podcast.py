import torch
from torch.utils.data import Dataset
import os
import torchaudio
import csv

MSP_SR = 16000  # MSP-Podcast is 16kHz in the current release

# Map MSP-Podcast primary categorical codes to your VoxProfile-like indices
# A: Angry, S: Sad, H: Happy, U: Surprise, F: Fear, D: Disgust, C: Contempt,
# N: Neutral, O: Other, X: No agreement

MSP_TO_VP_LABEL_MAPPING = {
    "A": 0,  # Angry      -> Anger
    "D": 2,  # Disgust    -> Disgust
    "F": 3,  # Fear       -> Fear
    "H": 4,  # Happy      -> Happiness
    "N": 5,  # Neutral    -> Neutral
    "S": 6,  # Sad        -> Sadness
    "U": 7,  # Surprise   -> Surprise
    "C": 1,  # Contempt   -> Contempt
    "O": 8,  # Other      -> Other
    "X": 8,  # No agreement -> Other (you can change this or filter later)
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

class MSPPodcastDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        split: str = "Train",  # "Train", "Development", "Test1", "Test2", "Test3"
        resample_rate: int = 16000,
        label_mapping: dict = MSP_TO_VP_LABEL_MAPPING,
        target_length_s: int = 5,
        audio_dir: str = "Audios",
        labels_subdir: str = "Labels",
        labels_filename: str = "labels_consensus.csv",
    ):
        """
        Minimal dataset for MSP-Podcast categorical emotions using labels_consensus.csv.

        Assumptions (adjust if needed):
            - Audio files:   <data_dir>/<audio_dir>/MSP-PODCAST_XXXX_YYYY.wav
            - Labels CSV:    <data_dir>/<labels_subdir>/<labels_filename>
            - CSV columns:   "FileName", "Emo_Prim", "Speaker", "Split_Set"
        """
        self.data_dir = data_dir
        self.split = split
        self.resample_rate = resample_rate
        self.target_length_s = target_length_s
        self.audio_dir = audio_dir
        self.label_mapping = label_mapping

        labels_path = os.path.join(data_dir, labels_subdir, labels_filename)

        # Load labels and split info from labels_consensus.csv
        # You may need to adjust these column names if your CSV differs.
        split_lower = split.strip().lower()
        self.sample_index = []

        with open(labels_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            fname_col   = "FileName"
            emo_col     = "EmoClass"
            spk_col     = "SpkrID"
            split_col   = "Split_Set"

            for row in reader:
                
                row_split = row[split_col].strip().lower()

                if row_split != split_lower: # Skip samples not in the desired split
                    continue

                fname = row[fname_col].strip()
                prim = row[emo_col].strip()
                spk = row[spk_col].strip()

                # Some files may not have a valid primary emotion (or mapping),
                # you can skip them here if desired.
                if prim not in self.label_mapping:
                    continue

                # Ensure extension
                if not fname.lower().endswith(".wav"):
                    fname = fname + ".wav"

                self.sample_index.append(
                    {
                        "fname": fname,
                        "spk": spk,
                        "prim": prim,
                    }
                )

        if resample_rate != MSP_SR:
            self.resample = torchaudio.transforms.Resample(MSP_SR, resample_rate)
        else:
            self.resample = None

        self.transcripts_dir = os.path.join(data_dir, "Transcripts")

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        sample_info = self.sample_index[idx]
        fname = sample_info["fname"]
        spk = sample_info["spk"]
        prim = sample_info["prim"]

        audio_path = os.path.join(
            self.data_dir,
            self.audio_dir,
            fname,
        )

        audio, sr = torchaudio.load(audio_path)
        assert sr == MSP_SR, f"Expected sample rate {MSP_SR}, but got {sr}"

        if self.resample is not None:
            audio = self.resample(audio)

        # Pad or truncate audio to target length (always take first target_length_s seconds)
        target_length_samples = int(self.target_length_s * self.resample_rate)
        audio_length = audio.size(1)

        if audio_length < target_length_samples:
            padding = target_length_samples - audio_length
            audio = torch.nn.functional.pad(audio, (0, padding))
            length = audio_length
        else:
            audio = audio[:, :target_length_samples]
            length = target_length_samples

        filename = os.path.splitext(fname)[0]

        transcript = open(
            os.path.join(self.transcripts_dir, f"{filename}.txt"), "r"
        ).read().strip()

        return {
            "audio": audio,
            "speaker": spk,
            "emotion": self.label_mapping[prim],
            "id": filename,
            "length": length,
            "filename": filename,
            "transcript": transcript,
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

        audio = torch.concat(padded, dim=0)  # (batch, channels, max_len) for mono
        emotions = torch.tensor([item["emotion"] for item in batch], dtype=torch.long)
        speakers = [item["speaker"] for item in batch]
        ids = [item["id"] for item in batch]
        filenames = [item["filename"] for item in batch]
        lengths = torch.tensor(lengths, dtype=torch.long)

        return {
            "audio": audio,
            "speaker": speakers,
            "emotion": emotions,
            "id": ids,
            "length": lengths,
            "filename": filenames,
            "transcript": [item["transcript"] for item in batch],
        }


if __name__ == "__main__":

    import tqdm

    # Example usage
    dataset = MSPPodcastDataset(
        data_dir="/data1/open_data/MSP-Podcast-1.11",
        split="Test1",
        resample_rate=16000,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=MSPPodcastDataset.collate_function
    )

    for batch in tqdm.tqdm(data_loader, total=len(data_loader)):
        print(batch["audio"].shape)
        print(batch["emotion"])
        print(batch["transcript"])