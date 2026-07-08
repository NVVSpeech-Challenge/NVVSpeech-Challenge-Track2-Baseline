# VoxCPM 扩词表指南

本指南介绍如何扩展 VoxCPM 模型的词表，添加自定义特殊 token（如情感标签、说话人标签等）或普通词汇。

## 使用方法

### 基本用法

```bash
python scripts/expand_vocabulary.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-expanded \
    --special_tokens "<|emotion_happy|>" "<|emotion_sad|>"
```

### 参数说明

| 参数 | 说明 | 必需 |
|------|------|------|
| `--pretrained_path` | 预训练模型路径 | 是 |
| `--output_path` | 输出路径 | 是 |
| `--special_tokens` | 特殊 token 列表 | 否 |
| `--regular_tokens` | 普通词汇 token 列表 | 否 |
| `--device` | 加载模型的设备（默认: cpu） | 否 |

### 示例

#### 1. 添加情感控制 token

```bash
python scripts/expand_vocabulary.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-emotion \
    --special_tokens "<|emotion_happy|>" "<|emotion_sad|>" "<|emotion_angry|>" "<|emotion_neutral|>"
```

#### 2. 添加说话人控制 token

```bash
python scripts/expand_vocabulary.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-speaker \
    --special_tokens "<|speaker_start|>" "<|speaker_end|>"
```

#### 3. 添加普通词汇

```bash
python scripts/expand_vocabulary.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-custom \
    --regular_tokens "自定义词1" "自定义词2" "专业术语"
```

#### 4. 同时添加特殊 token 和普通词汇

```bash
python scripts/expand_vocabulary.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-expanded \
    --special_tokens "<|emotion_happy|>" "<|emotion_sad|>" \
    --regular_tokens "新词1" "新词2"
```

## 输出文件

脚本会在输出目录生成以下文件：

- `tokenizer.json` - 更新后的 tokenizer
- `tokenizer_config.json` - tokenizer 配置
- `special_tokens_map.json` - 特殊 token 映射
- `custom_special_tokens.json` - 自定义特殊 token 列表
- `config.json` - 更新后的模型配置（vocab_size 已更新）
- `model.safetensors` 或 `pytorch_model.bin` - 更新后的模型权重
- `audiovae.pth` 或 `audiovae.safetensors` - AudioVAE 权重（从原始模型复制）

## 使用扩展后的模型

### 微调训练

更新训练配置文件，使用扩展后的模型路径：

```yaml
# conf/voxcpm_v1/voxcpm_finetune_all.yaml
pretrained_path: /path/to/VoxCPM-expanded  # 使用扩展后的模型路径
train_manifest: /path/to/train.jsonl
# ... 其他配置保持不变
```

### 推理

```python
from voxcpm.model import VoxCPMModel

# 加载扩展后的模型
model = VoxCPMModel.from_local("/path/to/VoxCPM-expanded")

# 使用特殊 token
text = "<|emotion_happy|>你好，今天天气真好！"
audio = model.generate(target_text=text)
```

## 注意事项

1. **Token ID 兼容性**
   - `audio_start_token` (101) 和 `audio_end_token` (102) 是硬编码的
   - 新添加的 token 会追加到词表末尾，不会影响这些特殊 token

2. **Embedding 初始化**
   - 新添加的 token 的 embedding 会使用正态分布随机初始化
   - 需要通过微调来学习这些新 token 的语义

3. **词表大小对齐**
   - 脚本会自动更新 `config.json` 中的 `vocab_size`
   - 确保与 tokenizer 的实际词表大小一致

4. **中文多字符 token**
   - 项目使用 `mask_multichar_chinese_tokens()` 函数
   - 新添加的中文多字符 token 可能会被拆分为单字符

5. **内存使用**
   - 扩展词表会增加 embedding 层的大小
   - 如果词表增加很多，可能需要更多内存

## 技术细节

### 扩词表流程

1. **加载 tokenizer** - 从预训练路径加载原始 tokenizer
2. **添加 token** - 将新 token 添加到 tokenizer 词表
3. **保存 tokenizer** - 保存更新后的 tokenizer 文件
4. **更新配置** - 更新 `config.json` 中的 `vocab_size`
5. **调整 embedding** - 调整模型 embedding 层大小
6. **保存模型** - 保存更新后的模型权重

### Embedding 调整策略

- 保留原始 embedding 权重
- 新 token 的 embedding 使用正态分布初始化（std=0.02）
- 这是 Transformer 模型常用的初始化策略

## 常见问题

### Q: 添加 token 后需要重新微调吗？

A: 是的。新添加的 token 的 embedding 是随机初始化的，需要通过微调来学习其语义。

### Q: 可以添加多少个 token？

A: 理论上没有限制，但建议：
- 根据实际需求添加
- 考虑内存和计算开销
- 过多的 token 可能影响模型性能

### Q: 如何验证 token 是否正确添加？

A: 可以使用以下代码验证：

```python
from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("/path/to/VoxCPM-expanded")
print(f"词表大小: {len(tokenizer)}")
print(f"特殊 token: {tokenizer.additional_special_tokens}")

# 测试分词
text = "<|emotion_happy|>你好"
tokens = tokenizer.tokenize(text)
print(f"分词结果: {tokens}")
```

### Q: 可以删除已添加的 token 吗？

A: 不建议直接删除 token，因为这会导致 embedding 层大小不匹配。如果需要移除 token，建议重新从原始模型开始扩展。

## 相关文件

- `scripts/expand_vocabulary.py` - 扩词表脚本
- `src/voxcpm/model/voxcpm.py` - VoxCPM 模型定义
- `src/voxcpm/model/utils.py` - 模型工具函数
- `src/voxcpm/modules/minicpm4/config.py` - MiniCPM4 配置
