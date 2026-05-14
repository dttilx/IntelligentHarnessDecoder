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

## 输出

默认输出到 `output/`：

- `output/pages/`：PDF 渲染出的页面图片
- `output/tiles/`：切片图片
- `output/ocr_raw/ocr_raw.csv`：OCR 原始识别文本
- `output/components.csv`：元器件候选明细
- `output/components.xlsx`：Excel 版结果，包含明细和去重名称
- `output/components.txt`：去重后的元器件名称列表
- `output/marked_pages/`：带识别框的校验图片

## 准确率和召回率优化

当前版本采用“先尽量识别，再规则清洗”的策略提升效果：

- PDF 页面先按较高 DPI 渲染，并进行灰度化、自动对比度增强、锐化和切片 OCR，尽量提高小字、密集标注和局部元器件名称的召回率。
- OCR 原始文本会完整保存到 `output/ocr_raw/ocr_raw.csv`，便于后续人工复核和继续调规则。
- 元器件提取阶段优先匹配常见名称后缀，例如控制器、传感器、继电器、保险、插接器、端子、搭铁、电线束、电磁阀、电机、开关、按钮、仪表、模块、灯、泵、阀等。
- 对常见 OCR 误识别做归一化修正，例如 `QBD诊断插座` 会修正为 `OBD诊断插座`，`刺叭`/`剌叭` 会修正为 `喇叭`，`问歇` 会修正为 `间歇`。
- 过滤技术说明类文本，例如包含测量点、观测方向、分布位置、保持力、工作温度、热缩管、压接、选装关系等内容的长句，减少把说明文字误当成元器件名称。
- 对线号、页码、纯数字、过长文本和低置信度文本进行过滤，并按页面、名称和位置做去重，减少重复框和噪声项。

这版更适合作为自动提取后的初筛结果：相比只用关键词截取，候选名称会更干净；相比只保留高置信度 OCR，又能保留更多真实元器件名称。若要继续提高准确率，建议基于 `components.xlsx` 建立一份人工标注标准答案，再按准确率、召回率、F1 分数迭代规则。

## 说明

这类 PDF 里的主体文字通常不是普通可复制文本，所以程序主要依赖 OCR。当前版本在召回率和准确率之间做了平衡：先通过切片 OCR 尽量识别完整文本，再通过元器件后缀、常见错字修正、说明文字过滤和去重规则输出较干净的候选名称。
