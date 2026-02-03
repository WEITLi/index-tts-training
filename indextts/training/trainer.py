"""
Trainer Class for Emotion Adapter Fine-tuning with Full AR Loss

Stage 2 Training implements:
1. AR Loss: Train emotion adapter to preserve speech synthesis quality
2. Adversarial Loss: Train emotion adapter to be speaker-invariant (via GRL)

Combined Loss: L = L_AR - alpha * L_Adv
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from indextts.training.utils import save_adapter_checkpoint


class AdapterTrainer:
    """
    Trainer for emotion adapter fine-tuning with GRL-based disentanglement.
    
    Implements Stage 2 training from IndexTTS paper:
    - Freeze GPT backbone + Speaker Perceiver  
    - Train Emotion Perceiver with AR loss + Adversarial loss
    - Use GRL to make emotion embedding speaker-invariant
    
    Loss function: L = L_AR - alpha * L_Adv
    
    Where:
    - L_AR = Cross-entropy loss for mel token prediction (preserves synthesis quality)
    - L_Adv = Speaker classification loss on emotion embedding (drives disentanglement)
    
    Args:
        model: UnifiedVoice model
        speaker_classifier: SpeakerClassifier for adversarial training  
        feature_extractor: FeatureExtractor for W2V-BERT features and mel codes
        tokenizer: TextTokenizer for text tokenization
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler (optional)
        device: Target device
        alpha: Adversarial loss weight (default: 0.1)
        ar_loss_weight: AR loss weight (default: 1.0)
        grad_clip: Max gradient norm for clipping
    """
    def __init__(
        self,
        model,
        speaker_classifier,
        feature_extractor,
        tokenizer,
        optimizer,
        scheduler=None,
        device='cuda',
        alpha=0.1,
        ar_loss_weight=1.0,
        grad_clip=1.0,
    ):
        self.model = model
        self.speaker_classifier = speaker_classifier
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.alpha = alpha
        self.ar_loss_weight = ar_loss_weight
        self.grad_clip = grad_clip
        
        # Metrics
        self.train_history = []
    
    def _tokenize_texts(self, texts):
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            text_tokens: (B, L) tensor of token IDs
            text_lengths: (B,) tensor of valid lengths
        """
        batch_tokens = []
        batch_lengths = []
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
            batch_tokens.append(tokens)
            batch_lengths.append(len(tokens))
        
        # Pad to same length
        max_len = max(batch_lengths)
        padded_tokens = []
        for tokens in batch_tokens:
            pad_len = max_len - len(tokens)
            if pad_len > 0:
                tokens = F.pad(tokens, (0, pad_len), value=self.model.stop_text_token)
            padded_tokens.append(tokens)
        
        text_tokens = torch.stack(padded_tokens)
        text_lengths = torch.tensor(batch_lengths, dtype=torch.long, device=self.device)
        
        return text_tokens, text_lengths
    
    def train_one_epoch(self, dataloader, epoch=0):
        """
        Train one epoch with AR loss + Adversarial loss.
        
        Stage 2 Loss: L = ar_weight * L_AR - alpha * L_Adv
        
        Args:
            dataloader: DataLoader instance
            epoch: Current epoch number
            
        Returns:
            metrics: Dict of training metrics
        """
        self.model.train()
        self.speaker_classifier.train()
        
        total_loss = 0
        total_ar_loss = 0
        total_adv_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Move to device
            target_audio = batch['target_audio'].to(self.device)
            target_audio_len = batch['target_audio_len'].to(self.device)
            speaker_ref_audio = batch['speaker_ref_audio'].to(self.device)
            speaker_ref_len = batch['speaker_ref_len'].to(self.device)
            emotion_ref_audio = batch['emotion_ref_audio'].to(self.device)
            emotion_ref_len = batch['emotion_ref_len'].to(self.device)
            speaker_ids = batch['speaker_id'].to(self.device)
            texts = batch['text']
            
            # =========== Extract Features ===========
            # Speaker features (frozen, no gradient)
            with torch.no_grad():
                spk_features = self.feature_extractor.extract(speaker_ref_audio)
                spk_feat_lengths = self.feature_extractor.get_feature_lengths(speaker_ref_len)
                
                # Get speaker conditioning latent (frozen)
                spk_cond_latent = self.model.get_conditioning(
                    spk_features.transpose(1, 2),
                    spk_feat_lengths
                )
            
            # Emotion features (trainable)
            emo_features = self.feature_extractor.extract(emotion_ref_audio)
            emo_feat_lengths = self.feature_extractor.get_feature_lengths(emotion_ref_len)
            
            # Extract mel codes from target audio (for AR loss supervision)
            with torch.no_grad():
                mel_codes, mel_lengths = self.feature_extractor.extract_mel_codes(
                    target_audio, target_audio_len
                )
            
            # Tokenize texts
            text_tokens, text_lengths = self._tokenize_texts(texts)
            
            # =========== Forward Through Trainable Emotion Adapter ===========
            # Get emotion conditioning (trainable)
            emo_vec = self.model.get_emo_conditioning(
                emo_features.transpose(1, 2),
                emo_feat_lengths
            )
            
            # Project to model dim
            emo_vec_proj = self.model.emovec_layer(emo_vec)
            emo_vec_final = self.model.emo_layer(emo_vec_proj)
            
            # =========== Compute AR Loss ===========
            # Create use_speed tensor (required by forward)
            use_speed = torch.zeros(len(texts), dtype=torch.long, device=self.device)
            
            # Call model's forward to get mel logits
            # Note: We pass pre-computed speaker latent and emotion vector
            mel_logits = self.model(
                speech_conditioning_latent=spk_cond_latent,
                text_inputs=text_tokens,
                text_lengths=text_lengths,
                mel_codes=mel_codes,
                mel_codes_lengths=mel_lengths,
                emo_speech_conditioning_latent=emo_features,  # Will be ignored since emo_vec is passed
                cond_mel_lengths=spk_feat_lengths,
                emo_cond_mel_lengths=emo_feat_lengths,
                emo_vec=emo_vec_final,  # Use our computed emotion vector
                use_speed=use_speed,
                do_spk_cond=False,  # We already computed speaker conditioning
            )
            
            # Build target for AR loss (shift mel_codes for autoregressive prediction)
            mel_targets = F.pad(mel_codes, (0, 1), value=self.model.stop_mel_token)
            mel_targets = mel_targets[:, 1:]  # Shift right
            
            # Compute cross-entropy loss for mel prediction
            # mel_logits shape: (B, S, vocab_size), mel_targets shape: (B, S)
            ar_loss = F.cross_entropy(
                mel_logits.reshape(-1, mel_logits.shape[-1]) if mel_logits.dim() == 3 else mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.shape[1]),
                mel_targets.reshape(-1),
                ignore_index=self.model.stop_mel_token
            )
            
            # =========== Compute Adversarial Loss ===========
            # Speaker classifier with GRL (gradient reversal)
            speaker_logits = self.speaker_classifier(emo_vec_final)
            adv_loss = F.cross_entropy(speaker_logits, speaker_ids)
            
            # =========== Total Loss ===========
            # L = ar_weight * L_AR - alpha * L_Adv
            # Note: The minus sign is handled by GRL reversing gradients
            loss = self.ar_loss_weight * ar_loss + self.alpha * adv_loss
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    max_norm=self.grad_clip
                )
                torch.nn.utils.clip_grad_norm_(
                    self.speaker_classifier.parameters(), 
                    max_norm=self.grad_clip
                )
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_ar_loss += ar_loss.item()
            total_adv_loss += adv_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ar': f'{ar_loss.item():.4f}',
                'adv': f'{adv_loss.item():.4f}',
            })
        
        # Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Compute averages
        metrics = {
            'loss': total_loss / num_batches,
            'ar_loss': total_ar_loss / num_batches,
            'adv_loss': total_adv_loss / num_batches,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        self.train_history.append(metrics)
        return metrics
    
    def train(self, dataloader, epochs, save_dir=None, save_every=5):
        """
        Full training loop.
        
        Args:
            dataloader: DataLoader instance
            epochs: Number of epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
        """
        print(f"\n[Training] Starting Stage 2 training for {epochs} epochs...")
        print(f"  Samples: {len(dataloader.dataset)}")
        print(f"  Batch size: {dataloader.batch_size}")
        print(f"  AR loss weight: {self.ar_loss_weight}")
        print(f"  Adversarial alpha: {self.alpha}")
        
        for epoch in range(epochs):
            metrics = self.train_one_epoch(dataloader, epoch)
            
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"Loss: {metrics['loss']:.4f}, "
                  f"AR: {metrics['ar_loss']:.4f}, "
                  f"Adv: {metrics['adv_loss']:.4f}, "
                  f"LR: {metrics['lr']:.6f}")
            
            # Save checkpoint
            if save_dir is not None:
                if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
                    save_adapter_checkpoint(
                        self.model, 
                        save_dir, 
                        epoch + 1,
                        speaker_classifier=self.speaker_classifier,
                        optimizer=self.optimizer
                    )
        
        print("\n[Done] Training complete!")
        return self.train_history
    
    def set_grl_alpha(self, alpha):
        """Update GRL strength"""
        self.alpha = alpha
        self.speaker_classifier.set_grl_alpha(alpha)
