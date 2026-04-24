import torch
from torch.utils.data import Dataset
import os
import torchaudio
import csv
import tqdm

VOX1_SR = 16000  # VoxCeleb1 is typically 16kHz

# Map VoxCeleb1 gender codes to indices
# M: Male, F: Female
VOX1_GENDER_MAPPING = {
    "m": 0,  # Male
    "f": 1,  # Female
}

GENDER_LABELS = [
    'Male',      # 0
    'Female',    # 1
]


class Vox1Dataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        metadata_file: str = "vox1_meta.csv",
        resample_rate: int = 16000,
        gender_mapping: dict = VOX1_GENDER_MAPPING,
        audio_subdir: str = "vox1_dev_wav",
    ):
        """
        Dataset for VoxCeleb1 audio files with gender labels.

        Assumptions (adjust if needed):
            - Audio files:   <data_dir>/id*/voxceleb_*.wav or similar structure
            - Metadata CSV:  <metadata_path> with columns "VoxCelebID", "Gender", "Nationality", "Set"
            - Gender codes:  "m" or "f"

        Args:
            data_dir: Root directory of VoxCeleb1 data
            metadata_file: Name of metadata CSV file containing gender information
            resample_rate: Target sample rate for audio
            gender_mapping: Dictionary mapping gender codes to indices
            audio_subdir: Subdirectory name within speaker folders containing audio files
        """
        self.data_dir = data_dir
        self.resample_rate = resample_rate
        self.audio_subdir = audio_subdir
        self.gender_mapping = gender_mapping

        self.sample_index = []

        # Load metadata if provided
        metadata_path = os.path.join(data_dir, metadata_file)
        if metadata_path and os.path.exists(metadata_path):
            self._load_from_metadata(metadata_path)
        else:
            raise ValueError(f"Metadata path '{metadata_path}' does not exist. Please provide a valid metadata CSV file.")

        # Setup resampling if needed
        if resample_rate != VOX1_SR:
            self.resample = torchaudio.transforms.Resample(VOX1_SR, resample_rate)
        else:
            self.resample = None

    def _load_from_metadata(self, metadata_path: str):
        """Load samples and gender labels from metadata CSV."""
        with open(metadata_path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # Adjust column names as needed for your metadata format
                speaker_id = row.get("VoxCeleb1 ID").strip()
                gender = row.get("Gender").strip().lower()

                if not speaker_id or gender not in self.gender_mapping:
                    continue

                # Find audio files for this speaker
                speaker_dir = os.path.join(self.data_dir, self.audio_subdir, 'wav', speaker_id)
                
                if os.path.isdir(speaker_dir):
                    for subdir in os.listdir(speaker_dir):
                        subdir_path = os.path.join(speaker_dir, subdir)
                        if os.path.isdir(subdir_path):
                            for fname in os.listdir(subdir_path):
                                if fname.lower().endswith(".wav"):
                                    self.sample_index.append({
                                        "speaker_id": speaker_id,
                                        "filename": fname,
                                        "gender": gender,
                                        "path": os.path.join(subdir_path, fname),
                                    })

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        sample_info = self.sample_index[idx]
        speaker_id = sample_info["speaker_id"]
        fname = sample_info["filename"]
        gender = sample_info["gender"]

        audio_path = sample_info["path"]

        audio, sr = torchaudio.load(audio_path)
        assert sr == VOX1_SR, f"Expected sample rate {VOX1_SR}, but got {sr}"

        if self.resample is not None:
            audio = self.resample(audio)

        filename_without_ext = os.path.splitext(fname)[0]

        return {
            "audio": audio,
            "speaker": speaker_id,
            "gender": self.gender_mapping[gender],
            "id": filename_without_ext,
            "length": audio.size(1),
            "filename": fname,
            "speaker_id": speaker_id,
        }

    @staticmethod
    def collate_function(batch):
        """Collate function for DataLoader."""
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
        genders = torch.tensor([item["gender"] for item in batch], dtype=torch.long)
        speakers = [item["speaker"] for item in batch]
        speaker_ids = [item["speaker_id"] for item in batch]
        ids = [item["id"] for item in batch]
        filenames = [item["filename"] for item in batch]
        lengths = torch.tensor(lengths, dtype=torch.long)

        return {
            "audio": audio,
            "speaker": speakers,
            "speaker_id": speaker_ids,
            "gender": genders,
            "id": ids,
            "length": lengths,
            "filename": filenames,
        }


if __name__ == "__main__":

    import tqdm

    # Example usage
    dataset = Vox1Dataset(
        data_dir="/project2/shrikann_35/nmehlman/data/svpp-data/vox1",
        metadata_file="vox1_meta.csv",
        resample_rate=16000,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=Vox1Dataset.collate_function
    )

    for batch in tqdm.tqdm(data_loader, total=len(data_loader)):
        print(batch["audio"].shape)
        print(batch["gender"])
        print(batch["speaker"])
