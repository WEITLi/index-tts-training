"""
Feature Extractor for W2V-BERT Features and Semantic Codec
"""

import os
import torch
import torch.nn as nn
from transformers import SeamlessM4TFeatureExtractor
import safetensors.torch
from huggingface_hub import hf_hub_download


class FeatureExtractor(nn.Module):
    """
    Wrapper for W2V-BERT feature extraction and semantic codec.
    
    Extracts:
    1. Normalized semantic features (for conditioning)
    2. Semantic codes (mel codes) for AR loss
    
    Args:
        model_dir: Directory containing model config (for loading semantic codec config)
        device: Target device for computation
    """
    def __init__(self, model_dir, device):
        super().__init__()
        self.device = device
        self.model_dir = model_dir
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        
        # Load semantic model (W2V-BERT)
        from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
        from omegaconf import OmegaConf
        
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(model_dir, "wav2vec2bert_stats.pt")
        )
        self.semantic_model = self.semantic_model.to(device).eval()
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)
        
        # Load config for semantic codec
        cfg_path = os.path.join(model_dir, "config.yaml")
        cfg = OmegaConf.load(cfg_path)
        
        # Load semantic codec (for extracting mel codes)
        print("[FeatureExtractor] Loading semantic codec...")
        self.semantic_codec = build_semantic_codec(cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(self.semantic_codec, semantic_code_ckpt)
        self.semantic_codec = self.semantic_codec.to(device).eval()
        print(f"[FeatureExtractor] Semantic codec loaded from {semantic_code_ckpt}")
        
        # Freeze all models
        for param in self.semantic_model.parameters():
            param.requires_grad = False
        for param in self.semantic_codec.parameters():
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
        if audio_16k.dim() == 1:
            audio_16k = audio_16k.unsqueeze(0)
        
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
    
    @torch.no_grad()
    def extract_mel_codes(self, audio_16k, audio_lengths=None):
        """
        Extract semantic codes (mel codes) from audio for AR loss.
        
        This uses the semantic_codec to quantize W2V-BERT features into
        discrete token IDs that the GPT model predicts.
        
        Args:
            audio_16k: (B, T) tensor of 16kHz audio
            audio_lengths: (B,) tensor of audio lengths in samples (optional)
            
        Returns:
            mel_codes: (B, S) tensor of mel token IDs
            mel_lengths: (B,) tensor of valid lengths
        """
        if audio_16k.dim() == 1:
            audio_16k = audio_16k.unsqueeze(0)
        
        batch_size = audio_16k.shape[0]
        
        # First extract normalized features
        feat = self.extract(audio_16k)  # (B, T, 1024)
        
        # Quantize to semantic codes using the RepCodec
        # quantize() returns (semantic_code, feat_reconstructed)
        mel_codes, _ = self.semantic_codec.quantize(feat)  # (B, S)
        
        # Calculate lengths based on audio lengths
        if audio_lengths is not None:
            # ~50Hz output rate for 16kHz input
            mel_lengths = (audio_lengths.float() / 16000 * 50).long().to(self.device)
            # Ensure we don't exceed actual sequence length
            mel_lengths = torch.clamp(mel_lengths, max=mel_codes.shape[1])
        else:
            mel_lengths = torch.tensor([mel_codes.shape[1]] * batch_size, 
                                       dtype=torch.long, device=self.device)
        
        return mel_codes, mel_lengths
    
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
        return (audio_lengths.float() / sample_rate * 50).long()
