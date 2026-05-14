# 汽车线束图 PDF 元器件名称解析

这个项目用于从汽车线束图/电路图 PDF 中提取元器件、部件、插接器、端子、保险、继电器、控制器等名称，并导出为 TXT、CSV、Excel，同时生成带 OCR 框的人工校验图片。

## 环境安装

建议使用 Python 3.10 或 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

如果 `paddlepaddle` 安装失败，可以先单独安装适合当前系统的 PaddlePaddle，再安装其它依赖。

## 快速运行

先跑单页做验证：

```powershell
python -m src.main "C:\path\to\wiring-diagram.pdf" --pages 4 --dpi 300
```

跑全部页面：

```powershell
python -m src.main "C:\path\to\wiring-diagram.pdf" --dpi 300
```

启用 AI 视觉审核候选名称：

```powershell
$env:OPENAI_API_KEY="你的 API Key"
python -m src.main "C:\path\to\wiring-diagram.pdf" --dpi 300 --ai-vision-review
```

如果没有设置 `OPENAI_API_KEY`，程序不会调用外部 API，只会生成裁剪图和离线审核清单，方便人工查看。

已经跑过 OCR 后，复用缓存重新提取和审核：

```powershell
python -m src.main "C:\path\to\wiring-diagram.pdf" --pages 4 --reuse-ocr --no-marked-images
```

只复用已有候选结果执行视觉审核：

```powershell
python -m src.main "C:\path\to\wiring-diagram.pdf" --pages 4 --vision-only --ai-vision-review --ai-vision-max-names 10
```

## 输出

默认输出到 `output/`，按用途分目录：

- `output/final/gold_names.txt`：高准确率元器件名称列表
- `output/final/recall_boost_names.txt`：高召回补漏名称列表
- `output/final/components.txt`：规则层高可信名称列表
- `output/final/components_candidates.txt`：规则层疑似候选名称列表
- `output/final/rejected_names.txt`：打分后拒绝的名称，便于排查
- `output/review/components_review.xlsx`：按证据打分后的审核表，包含分数、层级、来源、页码和判断理由
- `output/review/draft_gold_names.xlsx`：Excel 版高可信名称和召回补漏名称
- `output/review/ai_review_prompt.md`：可交给 AI 继续清洗候选名称的审核提示词
- `output/review/ai_vision_review.xlsx`：AI 视觉审核结果，启用 `--ai-vision-review` 且设置 API key 后生成
- `output/final/ai_verified_names.txt`：AI 视觉确认后的名称，启用 `--ai-vision-review` 且设置 API key 后生成
- `output/ai_vision/vision_manifest.csv`：送审名称、裁剪图、页码和 OCR 原文清单
- `output/ai_vision/crops/`：候选名称附近的原图裁剪，用于 AI 视觉审核或人工复核
- `output/ai_vision/vision_review_cache.jsonl`：AI 视觉审核缓存，避免同一裁剪图重复调用 API
- `output/raw/components.csv`：元器件候选明细，包含 `accepted`/`candidate` 分层标记
- `output/raw/ocr_raw.csv`：OCR 原始识别文本
- `output/raw/pdf_text.csv`：PDF 原生文本
- `output/images/`：PDF 页面图、切片图和带识别框的校验图

## 准确率和召回率优化

当前版本采用“先尽量识别，再规则清洗”的策略提升效果：

- PDF 页面先按较高 DPI 渲染，并进行灰度化、自动对比度增强、锐化和切片 OCR，尽量提高小字、密集标注和局部元器件名称的召回率。
- OCR 原始文本会完整保存到 `output/ocr_raw/ocr_raw.csv`，便于后续人工复核和继续调规则。
- 元器件提取阶段优先匹配常见名称后缀，例如控制器、传感器、继电器、保险、插接器、端子、搭铁、电线束、电磁阀、电机、开关、按钮、仪表、模块、灯、泵、阀等。
- 对常见 OCR 误识别做归一化修正，例如 `QBD诊断插座` 会修正为 `OBD诊断插座`，`刺叭`/`剌叭` 会修正为 `喇叭`，`问歇` 会修正为 `间歇`。
- 过滤技术说明类文本，例如包含测量点、观测方向、分布位置、保持力、工作温度、热缩管、压接、选装关系等内容的长句，减少把说明文字误当成元器件名称。
- 对线号、页码、纯数字、过长文本和低置信度文本进行过滤，并按页面、名称和位置做去重，减少重复框和噪声项。
- 输出分为高可信结果和疑似候选结果：`components.txt` 保持干净，`components_candidates.txt` 保留可能需要人工补漏的编号和短名称。
- 对相邻 OCR 的短编号和元器件名称做近邻合并，合并结果只进入疑似候选层，用来补回被 OCR 拆开的名称和标识。
- 额外提取 PDF 原生文本证据，并对候选进行打分，生成 `components_review.xlsx` 和 `draft_gold_names.txt`，用于快速形成 AI 初版标准答案。
- 可选启用 AI 视觉审核：程序会为高可信和召回补漏名称裁剪原图局部区域，让视觉模型结合图片上下文判断接受、修正或拒绝候选，输出 `ai_verified_names.txt`。
- 调试时可以使用 `--reuse-ocr` 复用 `output/raw/ocr_raw.csv`，跳过 PDF 渲染、切片和 PaddleOCR；也可以用 `--vision-only` 只重跑视觉审核。
- AI 视觉审核默认最多送审 30 个候选，支持 `--ai-vision-max-names` 调整；每次请求默认 30 秒超时，支持 `--ai-vision-timeout` 调整。遇到额度不足或鉴权错误会停止后续请求。

这版更适合作为自动提取后的初筛结果：相比只用关键词截取，候选名称会更干净；相比只保留高置信度 OCR，又能保留更多真实元器件名称。若要继续提高准确率，建议基于 `components.xlsx` 建立一份人工标注标准答案，再按准确率、召回率、F1 分数迭代规则。

## 说明

这类 PDF 里的主体文字通常不是普通可复制文本，所以程序主要依赖 OCR。当前版本在召回率和准确率之间做了平衡：先通过切片 OCR 尽量识别完整文本，再通过元器件后缀、常见错字修正、说明文字过滤和去重规则输出较干净的候选名称。
