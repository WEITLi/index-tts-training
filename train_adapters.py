"""
IndexTTS Emotion Adapter Fine-tuning Script

This script implements Stage 2 training: Fine-tune Emotion Perceiver with GRL-based
speaker-emotion disentanglement while keeping GPT and Speaker Perceiver frozen.

Now with FULL AR Loss + Adversarial Loss:
- AR Loss: Ensures synthesized speech quality is preserved
- Adversarial Loss: Ensures emotion embedding is speaker-invariant

Usage:
    python train_adapters.py --data_path ./data.jsonl --epochs 10 --batch_size 2

Reference: IndexTTS paper - Stage 2 training with GRL for speaker-emotion disentanglement
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf

# IndexTTS imports
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.text.tokenizer import TextTokenizer
from indextts.text.preprocess import TextNormalizer

# Training module imports
from indextts.training import (
    EmotionAdapterDataset,
    collate_fn,
    SpeakerClassifier,
    FeatureExtractor,
    AdapterTrainer,
    configure_trainable_params,
    load_adapter_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description='IndexTTS Emotion Adapter Fine-tuning')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to JSONL training data')
    parser.add_argument('--model_dir', type=str, default='checkpoints', 
                        help='Model checkpoint directory')
    parser.add_argument('--cfg_path', type=str, default='checkpoints/config.yaml', 
                        help='Config file path')
    parser.add_argument('--output_dir', type=str, default='adapter_checkpoints', 
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, 
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help='Adversarial loss weight')
    parser.add_argument('--ar_loss_weight', type=float, default=1.0, 
                        help='AR loss weight')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Resume from checkpoint')
    parser.add_argument('--num_workers', type=int, default=2, 
                        help='DataLoader workers')
    parser.add_argument('--save_every', type=int, default=5, 
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"[Device] Using {device}")
    
    # Load config
    cfg = OmegaConf.load(args.cfg_path)
    
    # Load dataset
    print(f"\n[Dataset] Loading from {args.data_path}")
    dataset = EmotionAdapterDataset(args.data_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # Load model
    print("\n[Model] Loading UnifiedVoice...")
    model = UnifiedVoice(**cfg.gpt)
    gpt_path = f"{args.model_dir}/{cfg.gpt_checkpoint}"
    load_checkpoint(model, gpt_path)
    model = model.to(device)
    print(f"[Model] Loaded from {gpt_path}")
    
    # Load tokenizer (required for AR loss)
    print("\n[Tokenizer] Loading TextTokenizer...")
    bpe_path = os.path.join(args.model_dir, cfg.dataset["bpe_model"])
    normalizer = TextNormalizer(enable_glossary=True)
    normalizer.load()
    tokenizer = TextTokenizer(bpe_path, normalizer)
    print(f"[Tokenizer] Loaded from {bpe_path}")
    
    # Create speaker classifier
    speaker_classifier = SpeakerClassifier(
        input_dim=cfg.gpt.model_dim,
        hidden_dim=256,
        num_speakers=dataset.num_speakers,
        grl_alpha=1.0
    ).to(device)
    
    # Feature extractor (includes semantic codec for mel codes)
    print("\n[FeatureExtractor] Loading W2V-BERT and Semantic Codec...")
    feature_extractor = FeatureExtractor(args.model_dir, device)
    
    # Configure trainable parameters
    trainable_params = configure_trainable_params(model, speaker_classifier)
    
    # Optimizer
    optimizer = AdamW(
        [{'params': p['params']} for p in trainable_params], 
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_adapter_checkpoint(
            model, args.resume, 
            speaker_classifier=speaker_classifier,
            optimizer=optimizer
        ) + 1
        print(f"[Resume] Starting from epoch {start_epoch}")
    
    # Create trainer with tokenizer
    trainer = AdapterTrainer(
        model=model,
        speaker_classifier=speaker_classifier,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,  # Required for AR loss
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        alpha=args.alpha,
        ar_loss_weight=args.ar_loss_weight,
    )
    
    # Run training
    trainer.train(
        dataloader=dataloader,
        epochs=args.epochs,
        save_dir=args.output_dir,
        save_every=args.save_every,
    )


if __name__ == '__main__':
    main()
