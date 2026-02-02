"""
Models for Emotion Adapter Training

Contains:
- GradientReversalLayer: For speaker-emotion disentanglement
- SpeakerClassifier: Adversarial classifier for disentanglement
"""

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer function.
    Forward: Identity
    Backward: Negate gradients scaled by alpha
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) for adversarial training.
    
    During forward pass, the input is unchanged.
    During backward pass, the gradient is multiplied by -alpha.
    
    This forces the upstream network to learn features that are
    invariant to the downstream classifier's objective.
    
    Args:
        alpha: Scaling factor for gradient reversal (default: 1.0)
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    
    def set_alpha(self, alpha):
        """Adjust GRL strength (useful for curriculum learning)"""
        self.alpha = alpha


class SpeakerClassifier(nn.Module):
    """
    Speaker classifier for adversarial training.
    
    Used with GRL to force emotion embedding to be speaker-invariant.
    The GRL is applied before the classifier, so gradients flowing back
    to the emotion encoder will push it to learn speaker-agnostic features.
    
    Args:
        input_dim: Dimension of input features (emotion embedding)
        hidden_dim: Hidden layer dimension
        num_speakers: Number of speaker classes
        grl_alpha: Initial GRL strength
    """
    def __init__(self, input_dim, hidden_dim, num_speakers, grl_alpha=1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_speakers)
        )

    def forward(self, emotion_embedding):
        """
        Args:
            emotion_embedding: (B, D) emotion embedding from perceiver
        Returns:
            logits: (B, num_speakers) speaker classification logits
        """
        reversed_emb = self.grl(emotion_embedding)
        return self.classifier(reversed_emb)

    def set_grl_alpha(self, alpha):
        """Adjust GRL strength during training"""
        self.grl.set_alpha(alpha)
