# time_inference

从照片的 EXIF、文件名、文件夹名、图像相似度等多种来源推断拍摄时间，并将整个相册整理导出为结构化文件夹。

---

## 核心能力

- **多源时间推断**：按置信度从高到低依次尝试 EXIF、文件名正则、文件夹语义、相似图借时、文件系统时间，综合决策出最可靠的时间
- **动态模式挖掘**：从大量文件名中自动归纳未知命名规范，支持中文前缀、多段数字（如 `Screenshot_2019_0723_152344`）等复杂格式
- **去重**：支持字节级（sha1）、视觉级（phash）、语义级（CLIP）三档去重
- **格式修复**：检测扩展名与真实格式不符的文件（如 `.jpg` 实际是 PNG），转换后输出；MPO（3D 照片）被正确识别，不触发误报
- **Live Photo**：解压 `.livp`，提取静态图参与完整 pipeline
- **结构化导出**：输出分 `confident / review / duplicate / unsupported` 四个 bucket，保持原始目录结构；HTML summary 支持点击文件名直接打开原图
- **全程断点续传**：Scan 阶段有独立缓存，pipeline 有 checkpoint，writer 按 action 逐条更新状态，任意阶段中断后可继续
- **多线程**：Scan、pipeline、writer 均支持 `--workers` 并发

---

## 安装

**Python 3.10+**

### 必装

```bash
pip install Pillow piexif imagehash tqdm
```

### HEIC / HEIF 支持（iPhone 照片）

```bash
pip install pillow-heif
```

### CLIP 语义相似推断（可选，需要 GPU 效果更好）

```bash
pip install open-clip-torch torch torchvision
pip install faiss-cpu   # 或 faiss-gpu
```

---

## 快速开始

### 最简：分析 + 预览

```bash
python main.py /photos --dry-run --dump-json result.json
```

### 生成导出计划，人工审查后执行

```bash
# 第一步：分析，生成 plan 文件和 HTML summary
python main.py /photos --export-dir /export --export-plan

# 用浏览器打开 /export/*_summary.html，点击文件名审查每个决策

# 第二步：执行
python main.py --write-only --export-dir /export
```

### 推荐的完整命令（大量照片）

```bash
# 首次运行
python main.py /photos \
  --export-dir /export \
  --dedup phash \
  --cache --checkpoint \
  --workers 4 \
  --export-plan --export-write

# 中途中断后续传
python main.py /photos \
  --export-dir /export \
  --dedup phash \
  --cache --checkpoint \
  --workers 4 \
  --export-write
```

---

## Pipeline 执行流程

```
[可选] LivePhotoStage     .livp → 解压到临时目录
         ↓
ScanStage                 扫描文件，计算 sha1 / phash / 图像元信息
                          Scan Cache 自动启用：已扫描文件直接恢复，跳过 sha1/phash 重算
         ↓
PatternMiner              从文件名挖掘未知命名规范 → 动态 DatePattern
         ↓
ExifStage                 读取 EXIF，提取时间 / GPS / 设备信息
FilenameStage             文件名 + 文件夹名模式匹配
[可选] ClipStage          计算 CLIP embedding
         ↓
ResolverStage             从所有证据中决策最终时间（写入 time_result）
         ↓
[BatchStage] SimilarStage 相似图借时（需要 --clip）
[BatchStage] DedupStage   全量去重分组
         ↓
ExportPlanner             生成 *_plan.jsonl + *_summary.html
ExportWriter              执行文件操作（copy / convert / duplicate）
```

---

## 导出目录结构

```
export/
├── 20260519150830_photos_plan.jsonl      # 机器可读，Writer 断点续传用
├── 20260519150830_photos_summary.html    # 人工审查，含可点击文件超链接
│
├── confident/          时间可信，可直接使用
│   └── (原始目录结构)
│
├── review/             时间置信度低 / 有冲突 / 需要人工确认
│   └── (原始目录结构)
│       ├── photo.jpg
│       └── photo.jpg.review.txt      ← 说明进入 review 的具体原因
│
├── duplicate/          被识别为重复的文件（保留最优，其余移至此）
│   └── (原始目录结构)
│       ├── photo.jpg
│       └── photo.jpg.duplicate.txt  ← 说明与哪张图重复、重复类型
│
└── unsupported/        不支持的格式（原样复制，不处理）
    └── (原始目录结构)
```

> **等式保证**：`total files = confident + review + duplicate + unsupported`

---

## 时间证据来源与置信度

| 来源 | 置信度 | 备注 |
|---|---|---|
| EXIF DateTimeOriginal | 1.00 | 最权威，快门按下的瞬间 |
| EXIF DateTimeDigitized | 0.90 | 数字化时间（扫描件常用）|
| 文件名（精确到秒）| 0.90 | 如 `IMG_20200101_143022.jpg` |
| 文件名（精确到天）| 0.75 | 如 `WeiXin_20190821.jpg` |
| EXIF DateTime | 0.60 | 容易被编辑软件改写 |
| 相似图借时 | 0.45–0.75 | 依赖 CLIP 余弦相似度 |
| 文件夹名 | 0.20–0.60 | 人工整理，精度通常较低 |
| 文件系统 mtime | 0.20 | 最后兜底，极不可靠 |

置信度低于 `--confidence-threshold`（默认 0.6）的文件进入 `review/` bucket。

---

## 去重档位

| 档位 | 判断依据 | 适用场景 |
|---|---|---|
| `no` | 不去重 | 只需时间推断 |
| `sha1`（默认）| 字节完全一致 | 安全，覆盖完全相同的副本 |
| `phash` | 感知哈希汉明距离 ≤ 4 | 不同格式/压缩率的同一张图 |
| `clip` | 包含 phash，且标记肉眼相似 | 连拍识别，需要 `--clip` |

每档包含上一档效果：`clip ⊃ phash ⊃ sha1`

重复组内保留"最优"文件的评分标准（分越高越好）：

- +3.0 有 EXIF DateTimeOriginal
- +2.0 时间置信度 ≥ 0.9
- +1.0 时间置信度 ≥ 0.7
- +1.0 文件格式与扩展名匹配（无需转换）
- +0.5 有 GPS 信息
- −1.0 是截图
- 文件大小作为最终 tiebreaker

---

## 完整命令行参数

### 基础

| 参数 | 默认 | 说明 |
|---|---|---|
| `input_dir` | — | 扫描根目录（`--write-only` 时可省略）|
| `--dry-run` | — | 只推断，不写任何文件 |
| `--workers` | 4 | 并发线程数（1 = 串行）|
| `--confidence-threshold` | 0.6 | 低于此值进入 review |

### 导出

| 参数 | 说明 |
|---|---|
| `--export-dir` | 导出根目录（四个 bucket 的父目录）|
| `--export-plan` | 生成 plan 文件和 HTML summary |
| `--export-write` | 执行导出（需先有 plan 文件）|
| `--write-only` | 只运行 Writer，跳过所有分析（`input_dir` 可省略）|

### 缓存与断点续传

| 参数 | 默认 | 说明 |
|---|---|---|
| `--cache` | — | 启用 CLIP embedding 等跨运行缓存 |
| `--checkpoint` | — | 启用 pipeline 断点续传 |
| `--cache-dir` | `.cache` | 缓存文件存放目录 |

> **Scan Cache 始终自动启用**（存于 `cache-dir/scan_cache.jsonl`），无需额外参数。已扫描过的文件（路径 + 大小 + mtime 不变）直接恢复，跳过 sha1 和 phash 计算。

### 去重

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dedup` | `sha1` | 去重档位：`no` / `sha1` / `phash` / `clip` |
| `--dedup-phash-threshold` | 4 | phash 汉明距离阈值 |

### CLIP（可选）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--clip` | — | 启用 CLIP embedding + 相似图时间推断 |
| `--clip-model` | `ViT-B-32` | 模型（越大越准越慢，可选 `ViT-L-14`）|
| `--clip-min-score` | 0.85 | 余弦相似度阈值 |

### 文件过滤

| 参数 | 说明 |
|---|---|
| `--livp` | 解压 Apple Live Photo `.livp` |
| `--skip-ext` | 追加跳过的扩展名（`pdf,txt`）；`=pdf` 完全覆盖默认值 |

默认跳过（不建立 item，不统计）：`ds_store` `db` `ini` `nomedia` `json` `xml` `txt` `md` `pdf` `xmp` `thm`

### 调试

| 参数 | 说明 |
|---|---|
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `--log-file` | 同时写入日志文件 |
| `--dump-json` | 将所有 item 完整序列化为 JSON |
| `--no-mine` | 禁用 PatternMiner（文件名模式自动挖掘）|

---

## 支持的文件格式

### 图片（参与完整 pipeline）

`jpg` `jpeg` `png` `gif` `bmp` `tiff` `tif` `webp`  
`heic` `heif` `avif`  
`raw` `cr2` `cr3` `nef` `arw` `orf` `rw2` `dng`

> **MPO**（富士/索尼 3D 照片）：扩展名 `.jpg`，Pillow 识别为 `MPO` 格式，属于合法格式，不触发格式转换。

### 视频（建立 item，跳过图像分析）

`mp4` `mov` `avi` `mkv` `m4v` `3gp` `wmv` `flv` `ts`

### Live Photo（需 `--livp`）

`.livp` → 解压，优先提取 `.heic`，其次 `.jpg`，作为普通图片参与 pipeline

---

## 典型使用场景

### 场景 1：只看时间推断结果，不导出

```bash
python main.py /photos --dry-run --dump-json result.json --log-level INFO
```

### 场景 2：整理并导出，去掉重复文件

```bash
python main.py /photos \
  --dedup phash \
  --export-dir /export \
  --export-plan --export-write
```

### 场景 3：iPhone 照片（HEIC + Live Photo）

```bash
pip install pillow-heif

python main.py /iphone_backup \
  --livp \
  --dedup sha1 \
  --export-dir /export \
  --export-plan --export-write
```

### 场景 4：大量照片，全功能

```bash
# 首次（建立各类缓存）
python main.py /photos \
  --dedup phash \
  --clip --cache \
  --checkpoint \
  --workers 8 \
  --export-dir /export \
  --export-plan --export-write

# 增量运行或续传（scan cache 和 embedding cache 自动命中）
python main.py /photos \
  --dedup phash \
  --clip --cache \
  --checkpoint \
  --workers 8 \
  --export-dir /export \
  --export-plan --export-write
```

### 场景 5：只重新执行导出（已有 plan）

```bash
# plan 文件已存在，只执行文件操作
python main.py --write-only --export-dir /export

# 或等价写法（省略 input_dir 时自动进入 write-only 模式）
python main.py --export-write --export-dir /export
```

---

## 安全说明

- **原始文件只读**：所有分析操作不修改输入目录中的任何文件
- **导出目录保护**：Writer 启动时检查 `export-dir` 不是 `input-dir` 的子目录，防止误操作覆盖源文件
- **幂等导出**：Writer 复制/转换前检查目标文件是否已存在且内容一致（sha1 比对），一致则跳过，不会重复写入
- **格式转换保留元数据**：转换时优先从原始文件传递 EXIF bytes 和 ICC 色彩配置文件给 Pillow，尽量避免元数据丢失；RAW 格式不做转换，直接复制

---

## 项目结构

```
time_inference/
├── config.py                  全局配置（所有可调参数）
├── main.py                    程序入口，CLI 参数解析
│
├── core/
│   ├── context.py             全局运行上下文（config / logger / cache / models）
│   ├── evidence.py            时间证据数据类
│   ├── item.py                照片 Item（贯穿整个 pipeline 的数据载体）
│   ├── pipeline.py            Stage 执行顺序定义
│   ├── scheduler.py           运行调度（多线程、checkpoint 协调）
│   └── stage_base.py          Stage / BatchStage 抽象基类
│
├── stages/
│   ├── scan.py                扫描目录，生成 Item 列表（含 Scan Cache）
│   ├── exif.py                EXIF 提取与时间推断
│   ├── filename.py            文件名 / 文件夹名时间推断
│   ├── clip.py                CLIP embedding 计算
│   ├── similar.py             相似图时间推断（BatchStage）
│   ├── resolver.py            多证据决策，写入 time_result
│   ├── dedup.py               全量去重分组（BatchStage）
│   └── livephoto.py           .livp 解压预处理
│
├── models/
│   ├── __init__.py            ModelContainer（统一模型持有者）
│   └── clip.py                CLIP 模型封装（懒加载，自动选设备）
│
├── storage/
│   ├── cache.py               计算结果缓存（CLIP 等，跨运行持久化，按 sha1 索引）
│   ├── scan_cache.py          Scan 断点续传缓存（按 relpath+stat 索引）
│   ├── checkpoint.py          Pipeline 运行状态缓存（断点续传）
│   └── index.py               faiss 向量索引 + phash 哈希索引
│
├── utils/
│   ├── patterns.py            预置时间正则模式库（11 种格式）
│   └── miner.py               文件名模式挖掘器（结构驱动 + 多假设解析）
│
└── export/
    ├── planner.py             ExportAction 决策 + HTML summary 生成
    └── writer.py              文件操作执行（copy / convert / duplicate）
```

---

## 缓存文件说明

程序运行后会在 `--cache-dir`（默认 `.cache`）下生成以下文件：

| 文件 | 内容 | 何时清除 |
|---|---|---|
| `scan_cache.jsonl` | 已扫描文件的 sha1 / phash / 图像信息 | 手动清除（跨运行持续有效）|
| `clip.jsonl` | CLIP embedding 向量（按 sha1 索引）| 手动清除（更换模型时需清除）|
| `checkpoint.jsonl` | Pipeline 运行状态（哪些 item 已完成）| 任务成功完成后自动删除 |
| `livp_unpacked/` | .livp 解压出的临时图片 | Writer 完成后可手动删除 |