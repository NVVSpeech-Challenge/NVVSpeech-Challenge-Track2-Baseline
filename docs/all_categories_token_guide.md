# VoxCPM 所有非语言声音类别 Token 使用指南

本指南介绍如何将训练数据中的**所有**非语言声音类别（包括单独类别和组合类别）添加到模型词表中。

## 数据统计

从 `dataset/merged_train_data.jsonl` 中分析的结果：

- **总数据量**: 273,424 条
- **类别总数**: 1,344 个
  - 单独类别: 23 个 (247,218 条)
  - 组合类别: 1,321 个 (26,206 条)

### 单独类别 (23 个)

| 类别 | 数据量 | 占比 |
|------|--------|------|
| [呼吸] | 66,198 | 26.8% |
| [惊讶] | 31,761 | 12.8% |
| [笑声] | 29,191 | 11.8% |
| [叹气] | 15,899 | 6.4% |
| [咳嗽] | 15,212 | 6.2% |
| [确认] | 11,882 | 4.8% |
| [疑问] | 10,617 | 4.3% |
| [迟疑] | 9,897 | 4.0% |
| [吸鼻] | 7,420 | 3.0% |
| [哭声] | 5,595 | 2.3% |
| [清嗓] | 4,993 | 2.0% |
| [不满] | 4,744 | 1.9% |
| [倒吸气] | 4,142 | 1.7% |
| [哈欠] | 3,997 | 1.6% |
| [嘘声] | 3,811 | 1.5% |
| [呻吟] | 3,577 | 1.4% |
| [吹口哨] | 3,381 | 1.4% |
| [打喷嚏] | 3,270 | 1.3% |
| [咂嘴] | 3,269 | 1.3% |
| [嘶声] | 3,148 | 1.3% |
| [哼唱] | 2,570 | 1.0% |
| [打鼾] | 2,464 | 1.0% |
| [鼓掌] | 180 | 0.1% |

### 组合类别 Top 20

| 类别 | 数据量 |
|------|--------|
| [呼吸],[笑声] | 825 |
| [呼吸],[惊讶] | 817 |
| [惊讶],[笑声] | 778 |
| [惊讶],[疑问] | 749 |
| [惊讶],[呼吸] | 739 |
| [呼吸],[确认] | 716 |
| [迟疑],[惊讶] | 594 |
| [迟疑],[笑声] | 560 |
| [呼吸],[疑问] | 463 |
| [迟疑],[呼吸] | 457 |
| [惊讶],[迟疑] | 442 |
| [叹气],[呼吸] | 430 |
| [迟疑],[确认] | 401 |
| [呼吸],[迟疑] | 376 |
| [惊讶],[确认] | 360 |
| [迟疑],[疑问] | 312 |
| [笑声],[确认] | 310 |
| [呼吸],[不满] | 301 |
| [笑声],[疑问] | 283 |
| [笑声],[惊讶] | 275 |

## 使用方法

### 1. 添加所有类别（推荐）

添加所有 1,344 个类别，每个类别会生成两种格式：
- 原始格式: `[笑声]` 或 `[呼吸],[笑声]`
- 前缀格式: `<|笑声|>` 或 `<|呼吸|笑声|>`

```bash
python scripts/add_all_category_tokens.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-all-categories \
    --data_path dataset/merged_train_data.jsonl
```

**注意**: 这将添加约 2,688 个 token（1,344 × 2），词表会显著增大。

### 2. 只添加高频类别

只添加出现次数 >= 10 的类别，减少词表膨胀：

```bash
python scripts/add_all_category_tokens.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-categories-min10 \
    --data_path dataset/merged_train_data.jsonl \
    --min_count 10
```

### 3. 只添加单独类别

不添加组合类别（如 `[呼吸],[笑声]`），只添加 23 个基础类别：

```bash
python scripts/add_all_category_tokens.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-single \
    --data_path dataset/merged_train_data.jsonl \
    --no_combo
```

### 4. 不添加前缀格式

只添加原始格式（如 `[笑声]`），不添加前缀格式（如 `<|笑声|>`）：

```bash
python scripts/add_all_category_tokens.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-categories \
    --data_path dataset/merged_train_data.jsonl \
    --no_prefix
```

### 5. 只生成报告

不添加 token，只生成类别分析报告：

```bash
python scripts/add_all_category_tokens.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/report \
    --data_path dataset/merged_train_data.jsonl \
    --report_only
```

## Token 格式说明

### 原始格式

与训练数据中的格式完全一致：

- 单独类别: `[笑声]`, `[呼吸]`, `[咳嗽]`
- 组合类别: `[呼吸],[笑声]`, `[惊讶],[疑问]`

### 前缀格式

更符合特殊 token 的通用约定，使用 `|` 分隔：

- 单独类别: `<|笑声|>`, `<|呼吸|>`, `<|咳嗽|>`
- 组合类别: `<|呼吸|笑声|>`, `<|惊讶|疑问|>`

## 输出文件

脚本会在输出目录生成：

- `tokenizer.json` - 更新后的 tokenizer
- `tokenizer_config.json` - tokenizer 配置
- `special_tokens_map.json` - 特殊 token 映射
- `config.json` - 更新后的模型配置
- `model.safetensors` - 更新后的模型权重
- `audiovae.pth` / `audiovae.safetensors` - AudioVAE 权重
- `category_analysis.txt` - 类别分析报告

## 使用扩展后的模型

### 训练配置

```yaml
# conf/voxcpm_v1/voxcpm_finetune_all.yaml
pretrained_path: /path/to/VoxCPM-all-categories
train_manifest: /path/to/train.jsonl
# ... 其他配置
```

### 推理示例

```python
from voxcpm.model import VoxCPMModel

model = VoxCPMModel.from_local("/path/to/VoxCPM-all-categories")

# 使用单独类别
text = "[笑声]你好，今天天气真好！"
audio = model.generate(target_text=text)

# 使用组合类别
text = "[呼吸],[笑声]你好，今天天气真好！"
audio = model.generate(target_text=text)

# 使用前缀格式
text = "<|笑声|>你好，今天天气真好！"
audio = model.generate(target_text=text)

# 使用组合类别前缀格式
text = "<|呼吸|笑声|>你好，今天天气真好！"
audio = model.generate(target_text=text)
```

## 推荐配置

### 方案 1: 完整添加（推荐用于研究）

添加所有类别，最大化模型能力：

```bash
python scripts/add_all_category_tokens.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-all-categories \
    --data_path dataset/merged_train_data.jsonl
```

- 优点：覆盖所有类别组合，模型能力最完整
- 缺点：词表大（约 +2,688 token），需要更多训练数据

### 方案 2: 高频类别（推荐用于生产）

只添加出现次数 >= 5 的类别：

```bash
python scripts/add_all_category_tokens.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-categories-min5 \
    --data_path dataset/merged_train_data.jsonl \
    --min_count 5
```

- 优点：词表适中，覆盖大部分常见场景
- 缺点：会丢失一些罕见组合

### 方案 3: 仅基础类别（推荐用于快速实验）

只添加 23 个单独类别：

```bash
python scripts/add_all_category_tokens.py \
    --pretrained_path /path/to/VoxCPM-0.5B \
    --output_path /path/to/VoxCPM-single \
    --data_path dataset/merged_train_data.jsonl \
    --no_combo
```

- 优点：词表增加最少（+46 token），训练快
- 缺点：无法处理组合类别

## 注意事项

1. **词表大小**
   - 添加所有类别会显著增加词表大小
   - 建议根据实际需求选择合适的 `min_count`

2. **Embedding 初始化**
   - 新 token 的 embedding 使用正态分布初始化（std=0.02）
   - 需要通过微调来学习这些 token 的语义

3. **训练数据格式**
   - 确保训练数据的 `asr` 字段包含正确的类别标记
   - `category` 字段用于统计，`asr` 字段用于训练

4. **组合类别的处理**
   - 组合类别 `[呼吸],[笑声]` 会被作为一个整体 token 添加
   - 不会被拆分为单独的 `[呼吸]` 和 `[笑声]`

5. **内存使用**
   - 添加大量 token 会增加 embedding 层的内存占用
   - 建议在内存充足的机器上运行

## 验证 Token 添加

```python
from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("/path/to/VoxCPM-all-categories")

# 检查词表大小
print(f"词表大小: {len(tokenizer)}")

# 检查特殊 token 数量
print(f"特殊 token 数量: {len(tokenizer.additional_special_tokens)}")

# 测试单独类别
text = "[笑声]你好"
tokens = tokenizer.tokenize(text)
print(f"单独类别分词: {tokens}")

# 测试组合类别
text = "[呼吸],[笑声]你好"
tokens = tokenizer.tokenize(text)
print(f"组合类别分词: {tokens}")

# 测试前缀格式
text = "<|呼吸|笑声|>你好"
tokens = tokenizer.tokenize(text)
print(f"前缀格式分词: {tokens}")
```

## 相关文件

- `scripts/add_all_category_tokens.py` - 添加所有类别的脚本
- `scripts/add_nonverbal_tokens.py` - 只添加基础类别的脚本
- `scripts/expand_vocabulary.py` - 通用扩词表脚本
- `dataset/merged_train_data.jsonl` - 训练数据
- `docs/all_categories_list.txt` - 所有类别完整列表
- `docs/nonverbal_tokens_guide.md` - 基础类别使用指南
