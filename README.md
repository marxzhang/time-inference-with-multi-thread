# time_inference

从照片的 EXIF、文件名、文件夹名、相似图等多种来源推断拍摄时间，并将照片整理导出到结构化目录。

## 功能概述

- **时间推断**：按可信度从高到低依次尝试 EXIF、文件名正则、文件夹名、相似图（CLIP）、文件系统时间
- **去重**：支持字节级（sha1）、感知哈希（phash）、语义相似（CLIP）三档去重强度
- **导出**：将照片分为 `confident` / `review` / `duplicate` / `unsupported` 四组输出，保持原始目录结构
- **Live Photo**：自动解压 `.livp` 文件，提取静态图参与流程
- **断点续传**：Scan 缓存 + Pipeline Checkpoint，中断后重启自动跳过已完成部分
- **多线程**：Pipeline 和 Writer 均支持并发，`--workers` 控制线程数

---

## 安装

```bash
# 基础依赖
pip install Pillow imagehash piexif tqdm

# HEIC/HEIF 支持（iPhone 照片）
pip install pillow-heif

# CLIP 相似图推断（可选，需要 GPU）
pip install open-clip-torch torch torchvision

# faiss 向量索引（CLIP 模式需要）
pip install faiss-cpu   # 或 faiss-gpu
```

Python 版本要求：>= 3.10

---

## 快速开始

```bash
# 最简运行：推断时间，不导出
python main.py /path/to/photos

# 推断 + 去重 + 生成导出计划
python main.py /path/to/photos \
  --dedup sha1 \
  --export-plan \
  --export-dir /path/to/export

# 审查 export_plan_summary.html 后执行导出
python main.py /path/to/photos \
  --export-write \
  --export-dir /path/to/export

# 完整流程（推断 + 去重 + 缓存 + 断点续传 + 一步导出）
python main.py /path/to/photos \
  --cache --checkpoint --workers 8 \
  --dedup phash \
  --export-plan --export-write \
  --export-dir /path/to/export
```

---

## 命令行参数

### 基础

| 参数 | 默认值 | 说明 |
|---|---|---|
| `input_dir` | （必填）| 扫描的根目录 |
| `--output`, `-o` | `""` | 输出目录，默认原地写回 |
| `--dry-run` | `false` | 只推断，不写回任何文件 |
| `--workers`, `-w` | `4` | 并发线程数 |
| `--log-level` | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `--log-file` | `""` | 日志写入文件路径，默认只输出终端 |

### 缓存与断点续传

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--cache` | `false` | 启用计算结果缓存（CLIP embedding 等跨运行复用） |
| `--checkpoint` | `false` | 启用 Pipeline 断点续传 |
| `--cache-dir` | `.cache` | cache / checkpoint 存放目录 |

> Scan 阶段的断点续传**始终启用**，不需要额外参数。

### 去重

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dedup` | `sha1` | 去重强度：`no` / `sha1` / `phash` / `clip`，每档包含上一档 |
| `--dedup-phash-threshold` | `4` | phash 汉明距离阈值（`phash` / `clip` 模式生效） |

去重强度说明：

- `no` — 不去重
- `sha1` — 字节完全相同（安全，适合清理明确副本）
- `phash` — 视觉相同，覆盖格式转换、JPEG 重压缩（阈值 ≤ 4 位）
- `clip` — 语义相似，覆盖连拍、同场景多张（需要 `--clip`）

### CLIP

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--clip` | `false` | 启用 CLIP embedding + 相似图时间推断 |
| `--clip-model` | `ViT-B-32` | 模型名称（`ViT-B-32` / `ViT-L-14` / `ViT-H-14`） |
| `--clip-min-score` | `0.85` | 相似图余弦相似度阈值 |

### 文件处理

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--livp` | `false` | 解压 `.livp`（Apple Live Photo），提取静态图参与流程 |
| `--skip-ext` | `""` | 追加跳过的扩展名（逗号分隔，不含点）；`=pdf,txt` 完全覆盖默认值 |
| `--no-mine` | `false` | 禁用文件名模式自动挖掘（PatternMiner） |

### 导出

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--export-dir` | `""` | 导出根目录 |
| `--export-plan` | `false` | 生成导出计划（`*_plan.jsonl` + `*_summary.html`） |
| `--export-write` | `false` | 执行导出计划中的文件操作，支持断点续传 |

### 调试

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--confidence-threshold` | `0.6` | 低于此值标记 `LOW_CONFIDENCE` |
| `--dump-json` | `""` | 将所有 item 结果导出为 JSON（调试用） |

---

## 时间推断来源与置信度

系统对每张照片收集多条 Evidence，由 ResolverStage 选出最优结果：

| 来源 | 置信度 | 精度 | 说明 |
|---|---|---|---|
| EXIF `DateTimeOriginal` | 1.0 | 秒 | 最权威，相机快门时刻 |
| EXIF `DateTimeDigitized` | 0.9 | 秒 | 数字化时刻，扫描件常用 |
| 文件名（含完整时间戳）| 0.9 | 秒 | 如 `IMG_20200101_143022.jpg` |
| 文件名（含日期）| 0.75 | 天 | 如 `WeiXin_20190821.jpg` |
| EXIF `DateTime` | 0.6 | 秒 | 易被编辑软件改写 |
| 文件夹名 | 0.3–0.7 | 年/月/天 | 人工整理的目录 |
| 相似图推断（CLIP）| 0.45–0.75 | 继承 | 借用相邻相似图的时间 |
| 文件系统 mtime | 0.2 | 秒 | 兜底，极不可靠 |

置信度低于 `--confidence-threshold`（默认 0.6）的照片会被标记为 `LOW_CONFIDENCE`，在导出时放入 `review/` 目录。

---

## 导出目录结构

```
export_dir/
├── 20240519143022_photos_plan.jsonl      # 导出计划（机器读取）
├── 20240519143022_photos_summary.html    # 导出摘要（人工审查，含超链接）
├── confident/                            # 时间可信，可直接使用
│   └── （原始目录结构）
├── review/                               # 需要人工确认
│   └── （原始目录结构）
│       ├── photo.jpg
│       └── photo.jpg.review.txt         # 说明原因
├── duplicate/                            # 去重后的冗余文件
│   └── （原始目录结构）
│       ├── photo.jpg
│       └── photo.jpg.duplicate.txt      # 说明与哪张图重复
└── unsupported/                          # 不支持的格式
    └── （原始目录结构）
```

`total = confident + review + duplicate + unsupported`

---

## 支持的格式

**图片**（完整分析）：
`jpg` `jpeg` `png` `gif` `bmp` `tiff` `tif` `webp` `heic` `heif` `avif` `raw` `cr2` `cr3` `nef` `arw` `orf` `rw2` `dng`

**视频**（生成 item，跳过图片字段分析）：
`mp4` `mov` `avi` `mkv` `m4v` `3gp` `wmv` `flv` `ts`

**特殊处理**：
- `.livp` — Apple Live Photo，需加 `--livp` 参数解压后处理

**默认跳过**（不计入统计）：
`ds_store` `db` `ini` `nomedia` `json` `xml` `txt` `md` `pdf` `xmp` `thm`

---

## 项目结构

```
time_inference/
├── main.py                  # 程序入口，CLI 参数解析
├── config.py                # 全局配置 dataclass
├── core/
│   ├── context.py           # 全局运行上下文（config、logger、cache、models）
│   ├── evidence.py          # 时间证据数据类
│   ├── item.py              # 照片 Item 数据类（pipeline 的数据载体）
│   ├── pipeline.py          # Stage 执行顺序定义
│   ├── scheduler.py         # 运行调度（支持多线程）
│   └── stage_base.py        # Stage 抽象基类
├── stages/
│   ├── scan.py              # 扫描目录，生成 Item 列表
│   ├── exif.py              # 提取 EXIF 时间、GPS、设备信息
│   ├── filename.py          # 从文件名 / 文件夹名推断时间
│   ├── clip.py              # 计算 CLIP embedding
│   ├── similar.py           # 相似图时间推断（BatchStage）
│   ├── dedup.py             # 去重分组（BatchStage）
│   ├── resolver.py          # 从多条 Evidence 决策最终时间
│   └── livephoto.py         # .livp 文件解压预处理
├── models/
│   ├── __init__.py          # ModelContainer（统一持有各 ML 模型）
│   └── clip.py              # CLIP 模型封装（懒加载，自动选择设备）
├── storage/
│   ├── cache.py             # 计算结果缓存（CLIP embedding 等，线程安全）
│   ├── scan_cache.py        # Scan 断点续传缓存（relpath+stat 寻址）
│   ├── checkpoint.py        # Pipeline 断点续传（item 级，线程安全）
│   └── index.py             # faiss 向量索引 + phash 索引
├── export/
│   ├── planner.py           # 导出规划器（生成 ExportAction 列表和 HTML summary）
│   └── writer.py            # 导出执行器（支持断点续传和多线程）
└── utils/
    ├── patterns.py          # 预置时间正则模式库
    ├── miner.py             # 文件名模式自动挖掘（字符类型模板统计）
    └── time.py              # 时间解析工具（预留）
```

---

## 典型使用场景

### 场景一：整理手机备份

```bash
python main.py ~/Pictures/iPhone_Backup \
  --livp \
  --dedup phash \
  --cache --checkpoint \
  --export-plan --export-write \
  --export-dir ~/Pictures/Organized
```

### 场景二：处理大量无 EXIF 的旧照片

```bash
# 第一步：分析推断（开启 CLIP 提高无 EXIF 照片的覆盖率）
python main.py /media/OldPhotos \
  --clip --cache \
  --dedup phash \
  --workers 8 \
  --export-plan \
  --export-dir /media/Export

# 审查 HTML summary，确认去重结果和 review 原因

# 第二步：执行导出
python main.py /media/OldPhotos \
  --export-write \
  --export-dir /media/Export
```

### 场景三：调试 / 查看推断结果

```bash
python main.py /path/to/photos \
  --dry-run \
  --log-level DEBUG \
  --dump-json /tmp/result.json
```

---

## 缓存机制说明

系统有三层缓存，各自解决不同问题：

| 缓存 | 存储位置 | Key | 解决的问题 |
|---|---|---|---|
| `scan_cache.jsonl` | `.cache/` | `relpath\|size\|mtime` | 避免重复计算 sha1 / phash |
| `clip.jsonl` 等 | `.cache/` | `sha1` | 避免重复计算 CLIP embedding |
| `checkpoint.jsonl` | `.cache/` | `item_id` | Pipeline 中断后续传 |

Scan 缓存**始终启用**，其余需要 `--cache` / `--checkpoint` 参数开启。

---

## 注意事项

- `--export-dir` 不能是 `input_dir` 的子目录，系统会在启动时检查
- `--export-write` 是幂等的：已完成的文件不会重复处理
- 格式转换（ext 与真实格式不符）会保留原始 EXIF；无 EXIF 时写入推断时间
- HEIC 格式需要安装 `pillow-heif`，否则该格式的 phash / 宽高等字段会为空
- RAW 格式（cr2 / nef 等）不做格式转换，直接复制
