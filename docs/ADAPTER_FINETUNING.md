# IndexTTS Emotion Adapter 微调指南

## 代码结构

```
indextts/training/
├── __init__.py           # 包导出
├── dataset.py            # EmotionAdapterDataset, collate_fn
├── models.py             # GradientReversalLayer, SpeakerClassifier
├── feature_extractor.py  # FeatureExtractor (W2V-BERT)
├── trainer.py            # AdapterTrainer
└── utils.py              # 参数配置、检查点保存/加载
```

## 快速开始

### 1. 准备数据 (JSONL 格式)

```json
{"audio_path": "/path/to/audio.wav", "text": "文本", "speaker_id": 0, "speaker_ref_audio": "/path/to/ref.wav"}
```

### 2. 运行训练

```bash
python train_adapters.py \
    --data_path ./your_data.jsonl \
    --epochs 10 \
    --batch_size 2 \
    --alpha 0.1
```

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | (必需) | JSONL 训练数据路径 |
| `--epochs` | 10 | 训练轮数 |
| `--batch_size` | 2 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--alpha` | 0.1 | GRL 对抗损失权重 |

### 4. 加载微调权重

```python
from indextts.training import load_adapter_checkpoint
from indextts.gpt.model_v2 import UnifiedVoice

model = UnifiedVoice(**cfg.gpt)
load_adapter_checkpoint(model, "adapter_checkpoints/adapter_epoch_10.pt")
```

## 训练原理 (GRL 解耦)

```
Emotion Audio → Emotion Perceiver → emo_vec
                                      ↓
                                     GRL (梯度反转)
                                      ↓
                               Speaker Classifier → Adversarial Loss
```

**目标**: 使 `emo_vec` 只含情感信息，不含说话人音色。

