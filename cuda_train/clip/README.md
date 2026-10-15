# OpenCLIP CUDA：CC3M ViT-B/32

## 目标与当前状态

直接使用 OpenCLIP 从零训练 ViT-B/32，使用完整 CC3M 图文训练集，不先做预训练模型
推理复现或 COCO Captions 小规模训练。此实验是 OpenCLIP + CC3M 训练复现，
不是 OpenAI 原始 4 亿图文数据训练结果的严格复现。

32 轮正式训练已完成：2026-10-14 02:54:49 至 14:57:12 UTC，含逐轮验证与
权重保存共约 12 小时 2 分钟。最终权重为
`logs/cc3m_vit_b32_8xa100/checkpoints/epoch_32.pt`。

`open_clip_torch[training]==3.0.0` 安装在现有公共环境
`cuda_train/venv-cuda-py312`（并非独立 CLIP 环境）。新增依赖见
[`requirements.txt`](requirements.txt)，实测环境快照见
[`results/environment.txt`](results/environment.txt)。

## 启动与恢复

从仓库根目录运行（要求 8 张空闲的 A100）：

```bash
bash cuda_train/clip/run_train.sh train
```

日志、验证指标和 checkpoint 写入 `cuda_train/clip/logs/cc3m_vit_b32_8xa100/`，
中断后仅在已保存 epoch checkpoint 的前提下使用
`bash cuda_train/clip/run_train.sh train --resume latest`。按 epoch 保存，
epoch 内的进度无法恢复；需要用不同名字试验时可在末尾加 `--name NEW_NAME`。

训练配方：OpenCLIP 3.0.0、ViT-B-32 **随机初始化**、完整训练集 2,905,954
样本、验证集 13,443 样本；8 卡 DDP，每卡 batch 64，全局对比 batch 512，
无梯度累积；32 epochs、AdamW（LR 5e-4、β₁ 0.9、β₂ 0.98、ε 1e-6、
weight decay 0.2）、2000 次 warmup 后 cosine LR，BF16 autocast，
2 workers/GPU，seed 0，每 epoch 验证并保存 checkpoint。OpenCLIP WebDataset
训练端按 worker 分片并取整 batch，本次每 epoch 实际采样 2,906,112 对。
本次使用 OpenCLIP 默认跨卡特征 gather（`local_loss=False`、
`gather_with_grad=False`），未启用 gradient checkpointing 或预训练权重。

## ImageNet-1K 零样本评测

从仓库根目录运行：

```bash
bash cuda_train/clip/run_eval.sh
```

默认评测最终的 epoch 32，未基于 ImageNet 标签挑选 checkpoint。仅用一张 GPU，
FP32、batch 256、8 workers，完整评测 ImageNet-1K 验证集的 50,000 张图片。
评测脚本复用 ViT 的 XML 标签读取器，并核对 1000 类 synset 顺序与
`/data/xumengying/models_and_datasets/LOC_synset_mapping.txt` 一致。
类别名使用 OpenCLIP 的 `IMAGENET_CLASSNAMES`，文本使用
`OPENAI_IMAGENET_TEMPLATES` 的 80 个模板；各模板文本特征归一化后求均值，
再次归一化，与归一化图像特征计算相似度。使用模型默认的验证预处理，
具体 resize、crop 与 normalize 写入结果 JSON。

可用 `--checkpoint PATH --output PATH` 评测其他权重；默认日志与结果位于
`results/imagenet_eval.log` 和 `results/imagenet_epoch32.json`。

2026-10-15 对最终 epoch 32 权重完成上述完整评测：

| 指标 | 结果 | 命中数 / 样本数 |
|---|---:|---:|
| ImageNet-1K zero-shot Top-1 | **13.762%** | 6,881 / 50,000 |
| ImageNet-1K zero-shot Top-5 | **29.444%** | 14,722 / 50,000 |

结果见 [评测 JSON（含 checkpoint SHA256）](results/imagenet_epoch32.json)，
原始输出见 [评测日志](results/imagenet_eval.log)。评测在一张 A100 40GB 上完成，
含加载模型、构建文本分类器、读取图片及计算 checkpoint SHA256 共约 154.5 秒，
不能将此耗时当作纯模型推理性能。本次结果是 CC3M 从零训练的结果，
不是 README 中官方 CLIP 预训练权重推理结果的替代值。

## CC3M 验证结果

最终 epoch 32 在完整 13,443 对验证集上的结果：

| 指标 | 图→文 | 文→图 |
|---|---:|---:|
| R@1 | 10.303% | 13.271% |
| R@5 | 23.551% | 28.498% |
| R@10 | 30.507% | 36.703% |

最终 CLIP val loss 为 2.2098；最低 val loss 为 epoch 18 的 1.3047。
最终权重并非最低验证 loss 权重，且这些检索指标不是 ImageNet 分类精度。

归档日志与配置：
- [正式训练日志](results/train.log)
- [完整训练参数](results/train_params.txt)
- [32 轮 CC3M 验证指标](results/cc3m_validation.jsonl)
- [数据结构扫描结果](results/data_validation.json)

`logs/` 保留原始训练目录与 checkpoint；`results/` 保存可提交的日志、配置及
评测结果。`logs/cc3m_vit_b32_smoke/` 是有效的有限 batch 试跑，不是正式训练结果；
`train_8xa100.log` 是控制台副本。两者保留本地、不重复提交。模型、数据与缓存不提交。

## 数据下载

使用 Hugging Face 的 `pixparse/cc3m-wds` WebDataset 快照：

- 固定 revision：`46f3d69f840e59d77d52e8decfe5baec97e94c7f`
- 训练：576 个 `cc3m-train-*.tar`，数据卡公布 2,905,954 个样本。
- 验证：16 个 `cc3m-validation-*.tar`，数据卡公布 13,443 个样本。
- 仓库公布总大小约 281GB，建议预留至少 400GB。
- 数据来源：`https://huggingface.co/datasets/pixparse/cc3m-wds`

在可访问 Hugging Face 的机器安装 `huggingface_hub`，然后执行：

```bash
hf download pixparse/cc3m-wds \
  --repo-type dataset \
  --revision 46f3d69f840e59d77d52e8decfe5baec97e94c7f \
  --include 'cc3m-train-*.tar' \
  --include 'cc3m-validation-*.tar' \
  --include 'README.md' --include '_info.json' \
  --local-dir /data/xumengying/models_and_datasets/cc3m_wds
```

若在另一台机器下载，可替换 `--local-dir`，完成后将整个目录拷贝到本机：
`/data/xumengying/models_and_datasets/cc3m_wds`。请保持文件名不变，不需要解压 tar。
如发现损坏分片或下载异常，应先人工补齐，不静默跳过分片后启动训练。

全量 tar 扫描确认训练 576 片、2,905,954 对，验证 16 片、13,443 对，
未发现坏 tar 或缺失 `.txt`。该检查不是所有图片逐张解码，也不是分片 SHA256 校验。

未来 MX-C500 复现应使用相同分片与训练配方。

当前已有 ImageNet-1K 验证数据，不需要再次下载。训练数据、模型权重和缓存
不提交 Git。数据卡提供的样本数不等同于独立哈希校验。
