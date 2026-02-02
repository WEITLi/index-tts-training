"""
Feature Extractor for W2V-BERT Features
"""

import os
import torch
import torch.nn as nn
from transformers import SeamlessM4TFeatureExtractor


class FeatureExtractor(nn.Module):
    """
    Wrapper for W2V-BERT feature extraction.
    
    Extracts normalized semantic features from 16kHz audio using
    the pretrained W2V-BERT model.
    
    Args:
        model_dir: Directory containing wav2vec2bert_stats.pt
        device: Target device for computation
    """
    def __init__(self, model_dir, device):
        super().__init__()
        self.device = device
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        
        # Load semantic model
        from indextts.utils.maskgct_utils import build_semantic_model
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(model_dir, "wav2vec2bert_stats.pt")
        )
        self.semantic_model = self.semantic_model.to(device).eval()
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)
        
        # Freeze semantic model
        for param in self.semantic_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract(self, audio_16k):
        """
        Extract W2V-BERT features from 16kHz audio.
        
        Args:
            audio_16k: (B, T) tensor of 16kHz audio
            
        Returns:
            features: (B, S, 1024) normalized features
        """
        # Handle both batched and single audio
        if audio_16k.dim() == 1:
            audio_16k = audio_16k.unsqueeze(0)
        
        # Move to CPU for feature extraction
        audio_np = audio_16k.cpu().numpy()
        
        inputs = self.extract_features(
            audio_np, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Use layer 17 hidden states
        feat = vq_emb.hidden_states[17]  # (B, T, 1024)
        
        # Normalize
        feat = (feat - self.semantic_mean) / self.semantic_std
        
        return feat
    
    def get_feature_lengths(self, audio_lengths, sample_rate=16000):
        """
        Compute output feature lengths given input audio lengths.
        
        Args:
            audio_lengths: (B,) tensor of audio sample counts
            sample_rate: Input sample rate (default 16000)
            
        Returns:
            feature_lengths: (B,) tensor of feature sequence lengths
        """
        # W2V-BERT has ~50Hz output rate for 16kHz input
        return (audio_lengths / sample_rate * 50).long()
