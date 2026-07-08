# VoxCPM 非语言声音 Token 使用指南

本指南介绍如何将训练数据中的非语言声音类别（如 [笑声]、[呼吸]、[咳嗽] 等）添加到模型词表中。

## 数据分析结果

从 `dataset/merged_train_data.jsonl` 中分析出的非语言声音类别：

| 类别 | 数据量 | 说明 |
|------|--------|------|
| [笑声] | 29,191 条 | 笑声 |
| [呼吸] | 66,198 条 | 呼吸声 |
| [惊讶] | 31,761 条 | 惊讶声 |
| [叹气] | 15,899 条 | 叹气声 |
| [咳嗽] | 15,212 条 | 咳嗽声 |
| [确认] | 11,882 条 | 确认声 |
| [疑问] | 10,617 条 | 疑问声 |
| [迟疑] | 9,897 条 | 迟疑声 |
| [吸鼻] | 7,420 条 | 吸鼻声 |
| [哭声] | 5,595 条 | 哭声 |
| [清嗓] | 4,993 条 | 清嗓声 |
| [不满] | 4,744 条 | 不满声 |
| [倒吸气] | 4,142 条 | 倒吸气声 |
| [哈欠] | 3,997 条 | 哈欠声 |
| [嘘声] | 3,811 条 | 嘘声 |
| [呻吟] | 3,577 条 | 呻吟声 |
| [吹口哨] | 3,381 条 | 口哨声 |
| [打喷嚏] | 3,270 条 | 喷嚏声 |
| [咂嘴] | 3,269 条 | 咂嘴声 |
| [嘶声] | 3,148 条 | 嘶嘶声 |
| [哼唱] | 2,570 条 | 哼唱声 |
| [打鼾] | 2,464 条 | 鼾声 |
| [鼓掌] | 180 条 | 鼓掌声 |

**总计**: 23 个基础类别，33,417 条包含 [笑声] 的数据（含组合类别）

## 使用方法

### 添加所有非语言类别

```bash
python scripts/add_nonverbal_tokens.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-nonverbal
```

### 添加特定类别

```bash
# 只添加笑声相关
python scripts/add_nonverbal_tokens.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-laugh \
    --categories "[笑声]"

# 添加多个类别
python scripts/add_nonverbal_tokens.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-expression \
    --categories "[笑声]" "[哭声]" "[叹气]"
```

### 不添加前缀格式

默认会同时添加两种格式：
- `[笑声]` - 原始格式
- `<|笑声|>` - 前缀格式

如果只需要原始格式：

```bash
python scripts/add_nonverbal_tokens.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-nonverbal \
    --no_prefix
```

## 输出文件

脚本会生成以下文件：

- `tokenizer.json` - 更新后的 tokenizer
- `config.json` - 更新后的模型配置
- `model.safetensors` - 更新后的模型权重
- `audiovae.pth` / `audiovae.safetensors` - AudioVAE 权重

## 使用扩展后的模型

### 训练配置

```yaml
# conf/voxcpm_v1/voxcpm_finetune_all.yaml
pretrained_path: /path/to/VoxCPM-nonverbal
train_manifest: /path/to/train.jsonl
# ... 其他配置
```

### 推理示例

```python
from voxcpm.model import VoxCPMModel

model = VoxCPMModel.from_local("/path/to/VoxCPM-nonverbal")

# 使用非语言 token
text = "[笑声]你好，今天天气真好！"
audio = model.generate(target_text=text)

# 或者使用前缀格式
text = "<|笑声|>你好，今天天气真好！"
audio = model.generate(target_text=text)
```

## 数据格式说明

训练数据中的 `asr` 字段包含非语言声音标记：

```json
{
  "asr": "[笑声]你好，今天天气真好！",
  "category": "[笑声]",
  "pure_asr": "你好，今天天气真好！",
  "events": [
    {
      "event_type": "[笑声]",
      "start_time": 0.0,
      "end_time": 0.5
    }
  ]
}
```

- `asr`: 包含非语言标记的完整文本
- `category`: 非语言类别（可能是组合，如 `[呼吸],[笑声]`）
- `pure_asr`: 纯文本（不含非语言标记）
- `events`: 非语言事件的时间戳信息

## 组合类别

训练数据中存在组合类别，如：
- `[呼吸],[笑声]` - 825 条
- `[惊讶],[笑声]` - 778 条
- `[迟疑],[笑声]` - 560 条

这些组合类别会被拆分为基础类别处理。例如 `[呼吸],[笑声]` 会被拆分为 `[呼吸]` 和 `[笑声]` 两个 token。

## 注意事项

1. **Token 格式选择**
   - `[笑声]`: 原始格式，与训练数据一致
   - `<|笑声|>`: 前缀格式，更符合特殊 token 的通用约定
   - 建议保留两种格式，提高灵活性

2. **Embedding 初始化**
   - 新添加的 token 会使用正态分布初始化（std=0.02）
   - 需要通过微调来学习这些 token 的语义

3. **词表大小**
   - 添加 23 个基础类别 + 23 个前缀格式 = 46 个新 token
   - 如果已存在部分 token，实际添加数量会减少

4. **训练数据准备**
   - 确保训练数据的 `asr` 字段包含正确的非语言标记
   - 可以使用 `pure_asr` 字段作为纯文本参考

## 验证 Token 添加

```python
from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("/path/to/VoxCPM-nonverbal")

# 检查词表大小
print(f"词表大小: {len(tokenizer)}")

# 检查特殊 token
print(f"特殊 token 数量: {len(tokenizer.additional_special_tokens)}")

# 测试分词
text = "[笑声]你好"
tokens = tokenizer.tokenize(text)
print(f"分词结果: {tokens}")

# 测试前缀格式
text = "<|笑声|>你好"
tokens = tokenizer.tokenize(text)
print(f"前缀格式分词: {tokens}")
```

## 相关文件

- `scripts/add_nonverbal_tokens.py` - 添加非语言 token 的脚本
- `scripts/expand_vocabulary.py` - 通用扩词表脚本
- `dataset/merged_train_data.jsonl` - 训练数据
- `docs/expand_vocabulary_guide.md` - 扩词表通用指南
