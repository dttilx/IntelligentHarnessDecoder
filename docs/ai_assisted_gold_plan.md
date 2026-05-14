# AI 辅助元器件清单提取目标

本分支用于实验“PDF 原生文本提取 + 候选打分 + AI 文本清洗”的推荐版方案，目标是在不要求人工从零整理标准答案的前提下，提高全局元器件名称提取质量。

## 目标

- 保留 `master` 分支当前稳定的 OCR + 规则提取流程。
- 在本分支新增自动化证据来源和评分机制，生成更可信的元器件清单。
- 输出一份 AI 初版标准答案，供人工快速审核，而不是让用户从零整理。

## 计划能力

- PDF 原生文本提取：使用 PyMuPDF 读取 PDF 内嵌文本，作为 OCR 之外的高置信来源。
- 候选打分：综合 OCR 置信度、来源类型、重复次数、命名模式、配置/说明文本特征，为每条候选生成分数。
- AI 文本清洗：先生成可交给 AI 审核的 `ai_review_prompt.md`，再由 AI 对候选名称进行 accepted/candidate/rejected 判断、名称归一化和理由说明。
- AI 视觉审核：可选开启 `--ai-vision-review`，为高可信和召回补漏名称裁剪原图局部区域，让视觉模型结合图片上下文判断候选是否真实、是否需要修正。
- 速度优化：支持 `--reuse-ocr` 复用已有 OCR CSV，支持 `--vision-only` 只重跑视觉审核；视觉审核带进度、缓存、短超时，并在额度不足或鉴权失败时停止后续请求。
- 国内模型接入：视觉审核支持阿里云百炼 DashScope OpenAI 兼容接口，可通过 `--ai-provider dashscope --ai-vision-model qwen3-vl-flash` 调用 Qwen3-VL-Flash。
- 审核输出：生成 `final/gold_names.txt`、`final/recall_boost_names.txt`、`review/components_review.xlsx` 和 `review/ai_review_prompt.md`；开启视觉审核后额外生成 `final/ai_verified_names.txt`、`final/final_answer_names.txt`、`review/final_answer_sources.xlsx`、`review/ai_vision_review.xlsx`、`ai_vision/vision_manifest.csv` 和 `ai_vision/crops/`。

## 非目标

- 不改变 `master` 分支已有稳定结果。
- 不要求用户人工标注完整标准答案。
- 不承诺无人工审核即可证明真实 90%+，但目标是让结果质量更接近 90%。
