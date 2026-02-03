"""
Dataset for Emotion Adapter Fine-tuning (with AR Loss support)
"""

import json
import torch
from torch.utils.data import Dataset
import torchaudio


class EmotionAdapterDataset(Dataset):
    """
    Dataset for emotion adapter fine-tuning with full AR Loss support.
    
    Expected JSONL format:
    {
        "audio_path": "/path/to/target_audio.wav",
        "text": "对应的文本内容",
        "speaker_id": 0,
        "speaker_ref_audio": "/path/to/speaker_reference.wav",
        "emotion_ref_audio": "/path/to/emotion_reference.wav"  # optional, defaults to audio_path
    }
    """
    def __init__(self, data_path, sample_rate=16000, max_audio_sec=15):
        self.data = []
        self.sample_rate = sample_rate
        self.max_audio_sec = max_audio_sec
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    self.data.append(item)
        
        # Collect unique speakers
        self.speaker_ids = list(set(item.get('speaker_id', 0) for item in self.data))
        self.speaker_to_idx = {spk: idx for idx, spk in enumerate(self.speaker_ids)}
        self.num_speakers = len(self.speaker_ids)
        
        print(f"[Dataset] Loaded {len(self.data)} samples, {self.num_speakers} speakers")

    def __len__(self):
        return len(self.data)

    def _load_audio(self, path):
        """Load and resample audio to target sample rate"""
        audio, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
        # Mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        # Truncate if too long
        max_samples = int(self.max_audio_sec * self.sample_rate)
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
        return audio.squeeze(0)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load target audio (for AR loss: used to extract mel codes)
        target_audio = self._load_audio(item['audio_path'])
        
        # Load speaker reference audio (only used for conditioning, frozen)
        speaker_ref_audio = self._load_audio(item['speaker_ref_audio'])
        
        # Load emotion reference audio (used for emotion conditioning, trainable)
        emotion_ref_path = item.get('emotion_ref_audio', item['audio_path'])
        emotion_ref_audio = self._load_audio(emotion_ref_path)
        
        # Speaker ID for adversarial loss
        speaker_id = self.speaker_to_idx[item.get('speaker_id', 0)]
        
        return {
            'target_audio': target_audio,
            'speaker_ref_audio': speaker_ref_audio,
            'emotion_ref_audio': emotion_ref_audio,
            'speaker_id': speaker_id,
            'text': item['text'],
        }


def collate_fn(batch):
    """Custom collate function to handle variable length audio"""
    max_target_len = max(item['target_audio'].shape[0] for item in batch)
    max_spk_len = max(item['speaker_ref_audio'].shape[0] for item in batch)
    max_emo_len = max(item['emotion_ref_audio'].shape[0] for item in batch)
    
    target_audios = []
    target_audio_lens = []
    speaker_ref_audios = []
    speaker_ref_lens = []
    emotion_ref_audios = []
    emotion_ref_lens = []
    speaker_ids = []
    texts = []
    
    for item in batch:
        # Pad target audio
        target_audio = item['target_audio']
        target_audio_lens.append(target_audio.shape[0])
        target_pad = torch.zeros(max_target_len - target_audio.shape[0])
        target_audios.append(torch.cat([target_audio, target_pad]))
        
        # Pad speaker ref
        spk_audio = item['speaker_ref_audio']
        speaker_ref_lens.append(spk_audio.shape[0])
        spk_pad = torch.zeros(max_spk_len - spk_audio.shape[0])
        speaker_ref_audios.append(torch.cat([spk_audio, spk_pad]))
        
        # Pad emotion ref
        emo_audio = item['emotion_ref_audio']
        emotion_ref_lens.append(emo_audio.shape[0])
        emo_pad = torch.zeros(max_emo_len - emo_audio.shape[0])
        emotion_ref_audios.append(torch.cat([emo_audio, emo_pad]))
        
        speaker_ids.append(item['speaker_id'])
        texts.append(item['text'])
    
    return {
        'target_audio': torch.stack(target_audios),
        'target_audio_len': torch.tensor(target_audio_lens, dtype=torch.long),
        'speaker_ref_audio': torch.stack(speaker_ref_audios),
        'speaker_ref_len': torch.tensor(speaker_ref_lens, dtype=torch.long),
        'emotion_ref_audio': torch.stack(emotion_ref_audios),
        'emotion_ref_len': torch.tensor(emotion_ref_lens, dtype=torch.long),
        'speaker_id': torch.tensor(speaker_ids, dtype=torch.long),
        'text': texts,
    }
