"""
Utility Functions for Adapter Training
"""

import os
import torch


def configure_trainable_params(model, speaker_classifier=None):
    """
    Freeze GPT + Speaker Perceiver, only train Emotion Adapter.
    
    Trainable modules:
    - emo_conditioning_encoder
    - emo_perceiver_encoder  
    - emovec_layer
    - emo_layer
    - speaker_classifier (if provided)
    
    Args:
        model: UnifiedVoice model
        speaker_classifier: Optional SpeakerClassifier for adversarial training
        
    Returns:
        trainable_params: List of parameter dicts for optimizer
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Define emotion adapter keywords
    trainable_keywords = [
        "emo_conditioning_encoder",
        "emo_perceiver_encoder",
        "emovec_layer",
        "emo_layer",
    ]
    
    # Unfreeze emotion adapter parameters
    trainable_params = []
    for name, param in model.named_parameters():
        if any(kw in name for kw in trainable_keywords):
            param.requires_grad = True
            trainable_params.append({'params': param, 'name': name})
    
    # Add speaker classifier parameters if provided
    if speaker_classifier is not None:
        for name, param in speaker_classifier.named_parameters():
            param.requires_grad = True
            trainable_params.append({
                'params': param, 
                'name': f'speaker_classifier.{name}'
            })
    
    # Print stats
    total_params = sum(p.numel() for p in model.parameters())
    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    classifier_trainable = 0
    if speaker_classifier is not None:
        classifier_trainable = sum(p.numel() for p in speaker_classifier.parameters())
    
    trainable_count = model_trainable + classifier_trainable
    
    print(f"\n[Parameters]")
    print(f"  Total model params: {total_params:,}")
    print(f"  Trainable (model): {model_trainable:,}")
    print(f"  Trainable (classifier): {classifier_trainable:,}")
    print(f"  Trainable total: {trainable_count:,}")
    print(f"  Trainable ratio: {trainable_count/total_params:.2%}")
    
    return trainable_params


def save_adapter_checkpoint(model, save_path, epoch, speaker_classifier=None, optimizer=None):
    """
    Save only the trainable adapter parameters.
    
    Args:
        model: UnifiedVoice model
        save_path: Directory to save checkpoint
        epoch: Current epoch number
        speaker_classifier: Optional classifier to save
        optimizer: Optional optimizer state to save
        
    Returns:
        save_file: Path to saved checkpoint
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Collect adapter state dict
    adapter_state = {}
    trainable_keywords = [
        "emo_conditioning_encoder",
        "emo_perceiver_encoder",
        "emovec_layer",
        "emo_layer",
    ]
    
    for name, param in model.named_parameters():
        if any(kw in name for kw in trainable_keywords):
            adapter_state[name] = param.data.cpu()
    
    # Build checkpoint
    checkpoint = {
        'epoch': epoch,
        'adapter_state_dict': adapter_state,
    }
    
    if speaker_classifier is not None:
        checkpoint['speaker_classifier_state_dict'] = speaker_classifier.state_dict()
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    save_file = os.path.join(save_path, f'adapter_epoch_{epoch}.pt')
    torch.save(checkpoint, save_file)
    print(f"[Checkpoint] Saved to {save_file}")
    
    return save_file


def load_adapter_checkpoint(model, checkpoint_path, speaker_classifier=None, optimizer=None):
    """
    Load adapter weights into model.
    
    Args:
        model: UnifiedVoice model
        checkpoint_path: Path to checkpoint file
        speaker_classifier: Optional classifier to load
        optimizer: Optional optimizer to load state
        
    Returns:
        epoch: Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    adapter_state = checkpoint['adapter_state_dict']
    
    # Load into model
    model_state = model.state_dict()
    for name, param in adapter_state.items():
        if name in model_state:
            model_state[name] = param
        else:
            print(f"[Warning] Key not found in model: {name}")
    
    model.load_state_dict(model_state)
    print(f"[Checkpoint] Loaded adapter weights from {checkpoint_path}")
    
    # Load classifier if provided
    if speaker_classifier is not None and 'speaker_classifier_state_dict' in checkpoint:
        speaker_classifier.load_state_dict(checkpoint['speaker_classifier_state_dict'])
        print(f"[Checkpoint] Loaded speaker classifier weights")
    
    # Load optimizer if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[Checkpoint] Loaded optimizer state")
    
    return checkpoint.get('epoch', 0)


def get_trainable_param_names(model):
    """Get list of trainable parameter names"""
    return [name for name, param in model.named_parameters() if param.requires_grad]


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
