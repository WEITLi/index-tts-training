"""
IndexTTS Training Module

This package contains modular components for emotion adapter fine-tuning.
"""

from indextts.training.dataset import EmotionAdapterDataset, collate_fn
from indextts.training.models import GradientReversalLayer, SpeakerClassifier
from indextts.training.feature_extractor import FeatureExtractor
from indextts.training.trainer import AdapterTrainer
from indextts.training.utils import (
    configure_trainable_params,
    save_adapter_checkpoint,
    load_adapter_checkpoint
)

__all__ = [
    'EmotionAdapterDataset',
    'collate_fn',
    'GradientReversalLayer',
    'SpeakerClassifier',
    'FeatureExtractor',
    'AdapterTrainer',
    'configure_trainable_params',
    'save_adapter_checkpoint',
    'load_adapter_checkpoint',
]
