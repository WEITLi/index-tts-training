"""
Trainer Class for Emotion Adapter Fine-tuning
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
    
    Implements Stage 2 training:
    - Freeze GPT + Speaker Perceiver
    - Train Emotion Perceiver with adversarial loss
    - Use GRL to make emotion embedding speaker-invariant
    
    Args:
        model: UnifiedVoice model
        speaker_classifier: SpeakerClassifier for adversarial training
        feature_extractor: FeatureExtractor for W2V-BERT features
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler (optional)
        device: Target device
        alpha: GRL loss weight
        grad_clip: Max gradient norm for clipping
    """
    def __init__(
        self,
        model,
        speaker_classifier,
        feature_extractor,
        optimizer,
        scheduler=None,
        device='cuda',
        alpha=0.1,
        grad_clip=1.0,
    ):
        self.model = model
        self.speaker_classifier = speaker_classifier
        self.feature_extractor = feature_extractor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.alpha = alpha
        self.grad_clip = grad_clip
        
        # Metrics
        self.train_history = []
    
    def train_one_epoch(self, dataloader, epoch=0):
        """
        Train one epoch.
        
        Loss = alpha * adversarial_loss
        (AR loss requires full forward pass with text/mel, simplified here)
        
        Args:
            dataloader: DataLoader instance
            epoch: Current epoch number
            
        Returns:
            metrics: Dict of training metrics
        """
        self.model.train()
        self.speaker_classifier.train()
        
        total_loss = 0
        total_adv_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Move to device
            speaker_ref_audio = batch['speaker_ref_audio'].to(self.device)
            emotion_ref_audio = batch['emotion_ref_audio'].to(self.device)
            speaker_ids = batch['speaker_id'].to(self.device)
            
            # Extract features (frozen)
            with torch.no_grad():
                spk_features = self.feature_extractor.extract(speaker_ref_audio)
                spk_lengths = torch.tensor(
                    [spk_features.shape[1]] * spk_features.shape[0], 
                    device=self.device
                )
            
            # Emotion features (need gradients for encoder)
            emo_features = self.feature_extractor.extract(emotion_ref_audio)
            emo_lengths = torch.tensor(
                [emo_features.shape[1]] * emo_features.shape[0], 
                device=self.device
            )
            
            # Forward through emotion adapter (trainable)
            with torch.enable_grad():
                emo_vec = self.model.get_emo_conditioning(
                    emo_features.transpose(1, 2),  # (B, 1024, S)
                    emo_lengths
                )  # (B, D)
                
                # Project to model dim
                emo_vec_proj = self.model.emovec_layer(emo_vec)
                emo_vec_final = self.model.emo_layer(emo_vec_proj)
            
            # Adversarial loss via speaker classifier (with GRL)
            speaker_logits = self.speaker_classifier(emo_vec_final)
            adv_loss = F.cross_entropy(speaker_logits, speaker_ids)
            
            # Total loss (adversarial only in this simplified version)
            loss = self.alpha * adv_loss
            
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
            total_adv_loss += adv_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'adv': f'{adv_loss.item():.4f}',
            })
        
        # Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Compute averages
        metrics = {
            'loss': total_loss / num_batches,
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
        print(f"\n[Training] Starting for {epochs} epochs...")
        print(f"  Samples: {len(dataloader.dataset)}")
        print(f"  Batch size: {dataloader.batch_size}")
        print(f"  GRL alpha: {self.alpha}")
        
        for epoch in range(epochs):
            metrics = self.train_one_epoch(dataloader, epoch)
            
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"Loss: {metrics['loss']:.4f}, "
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
