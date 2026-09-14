# VMMU: A Vietnamese Multitask Multimodal Understanding and Reasoning Benchmark

<div align="center">    
  <p style="font-size: 20px;">by 
    <a href="https://www.linkedin.com/in/dang-thi-tuong-vy-00a357278/">Vy Tuong Dang</a><sup>*1</sup>,
    <a href="https://anvo25.github.io/">An Vo</a><sup>*1,2</sup>,
    <a href="http://villacu.github.io/">Emilio Villa-Cueva</a><sup>2</sup>,
    <a href="https://www.linkedin.com/in/quang-tau-a708b4238/?originalSubdomain=kr">Quang Tau</a><sup>1</sup>, 
    <a href="https://www.resl.kaist.ac.kr/members/master-student#h.fiaa4al7sz8u">Duc Dm</a><sup>1</sup>, 
    <a href="https://mbzuai.ac.ae/study/faculty/thamar-solorio/">Thamar Solorio</a><sup>2</sup>,
    <a href="https://www.resl.kaist.ac.kr/members/director">Daeyoung Kim</a><sup>1</sup>
  </p>

  <p>
    <sup>*</sup>Equal contribution<br>
    <sup>1</sup>KAIST &nbsp;&nbsp;
    <sup>2</sup>MBZUAI
  </p>

[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://vmmu-bench.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.13680-b31b1b.svg)](https://arxiv.org/abs/2508.13680)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/anvo25/vmmu)
[![Code License](https://img.shields.io/badge/Code_License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Abstract

<p align="center">
  <img src="images/question_image.jpg" alt="VMMU Dataset Overview" width="80%"/>
</p>

*Vision language models are predominantly evaluated on English-centric multimodal benchmarks, leaving their behavior in other languages and cultures insufficiently understood. We introduce **VMMU**, a Vietnamese Multitask Multimodal Understanding and Reasoning benchmark of **2,548 multiple-choice questions** across seven domains: Mathematics, Physics, Chemistry, Biology, Geography, Driving Test, and IQ Test. Every question requires jointly reasoning over Vietnamese text and non-text visual evidence (charts, diagrams, tables, traffic scenes), and is distributed as a single rendered page image. We evaluate 10 open-source and 5 closed-source VLMs on VMMU. Test-time compute improves performance from **57%** to **78%** accuracy, but it remains inferior to an expert baseline (**99%** accuracy). Extensive error analyses show that closed-source VLMs already achieve strong Vietnamese OCR performance, yet still struggle on VMMU. This suggests that the primary bottleneck is **multimodal grounding and reasoning** rather than text recognition or Vietnamese language understanding. The dataset is available on [Hugging Face](https://huggingface.co/datasets/anvo25/vmmu).*

---

## 👋 Trying out our questions on your model

Use these challenging Vietnamese multimodal exam questions where most tested models fail to answer correctly. Each question requires understanding both Vietnamese text and visual elements like diagrams, charts, and illustrations.

---

## 🧪 Dataset Overview

| Subject      | #Questions |
| ------------ | ---------- |
| Mathematics  | 456        |
| Physics      | 361        |
| Chemistry    | 302        |
| Biology      | 341        |
| Geography    | 481        |
| Driving Test | 367        |
| IQ Test      | 240        |
| **Total**    | **2,548**  |

> Each question is an image containing both Vietnamese text and visuals. Most are 4-option multiple-choice questions. No screenshots of text-only questions are included — all questions are genuinely **multimodal**.

## 🚀 Quick Start Guide

### Option 1: Use Pre-built Dataset (Recommended for evaluating your models)

**If you just want to evaluate VLMs on our Vietnamese exam questions:**

🔥 **Download the complete dataset from Hugging Face** with full images and annotations:
- Go to our [Hugging Face dataset](https://huggingface.co/datasets/anvo25/vmmu)
- Download ready-to-use Vietnamese multimodal exam questions

This is the fastest way to get started with evaluation.

### Option 2: Reproduce/Generate Dataset

**If you want to reproduce our data pipeline or create custom variations:**

Please follow the installation and generation steps below to run the complete pipeline locally.

---

## 💻 Getting Started

```bash
git clone https://github.com/vytuongdang/VMMU.git
cd VMMU
pip install -r requirements.txt
```

## 📊 Tasks

VMMU spans **7 distinct domains** representative of Vietnamese educational assessments:

### Academic Subjects (Tasks 1-5)
- **Mathematics**: Function analysis, calculus, geometry (456 questions)
- **Physics**: Mechanics, waves, thermodynamics (361 questions) 
- **Chemistry**: Organic chemistry, electrochemistry (302 questions)
- **Biology**: Genetics, molecular biology (341 questions)
- **Geography**: Data visualization, economic geography (481 questions)

### Practical Assessments (Tasks 6-7)
- **[Driving Test](dataset/question_image/driving/)**: Traffic rules, road signs, safety scenarios (367 questions)
- **[IQ Test](dataset/question_image/iq/)**: Pattern recognition, logical reasoning (240 questions)

*All questions integrate Vietnamese text with visual elements (diagrams, charts, illustrations) at multiple resolutions.*

---

## 🚀 Quickstart

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Add your API key

Set the environment variable of the provider you use (or put the key in its file under `api_key/`):

| Provider | Environment variable | Key file | Models |
| --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `api_key/openai_key.txt` | `gpt-*`, `o3-*` |
| Anthropic | `ANTHROPIC_API_KEY` | `api_key/claude_key.txt` | `claude-*` |
| OpenRouter | `OPENROUTER_API_KEY` | `api_key/openrouter_key.txt` | `google/gemini-*`, `qwen/*`, `meta-llama/*`, `mistralai/*`, ... |
| Google | `GEMINI_API_KEY` | `api_key/gemini_key.txt` | `gemini-*` with `--provider gemini` |
| Cohere | `COHERE_API_KEY` | `api_key/cohere_key.txt` | `c4ai-aya-vision-*` |

### 3. Run evaluation on VLMs

The dataset is downloaded from [Hugging Face](https://huggingface.co/datasets/anvo25/vmmu) automatically and the API provider is picked from the model name, so only `--model` changes between models:

```bash
# Evaluate single model
python api_code/main_api.py \
  --model o3-2025-04-16 \
  --split full_vqa \
  --prompt_language vn

python api_code/main_api.py --model claude-sonnet-4-20250514 --split full_vqa --prompt_language vn
python api_code/main_api.py --model google/gemini-2.5-flash --split full_vqa --prompt_language vn
python api_code/main_api.py --model qwen/qwen2.5-vl-72b-instruct --split full_vqa --prompt_language vn

# Cross-lingual evaluation
python api_code/main_api.py \
  --model gpt-4.1-2025-04-14 \
  --split full_vqa \
  --prompt_language en

# Quick test on the first 5 questions
python api_code/main_api.py --model gpt-4.1-2025-04-14 --split random_subset_vqa --test 5
```

Available splits: `full_vqa`, `random_subset_vqa`, `random_subset_ocr`, `cropped_random_subset_vqa`, `cropped_random_subset_vqa_description` (the `cropped_*` splits only have Vietnamese prompts).

Results are saved to `results/<model>/<split>_<prompt_language>.json`; running the same command again only retries missing questions. When the run finishes, the accuracy is printed right away:

```
o3-2025-04-16 | full_vqa_vn.json | 2548 questions
|Subject     |    N| Extraction| Accuracy|
|------------|-----|-----------|---------|
|Math        |  456|        ...|      ...|
|...         |     |           |         |
|Overall     | 2548|        ...|      ...|
|Mean        |     |        ...|      ...|
```

Other options: `--provider` (force `openai` / `claude` / `gemini` / `openrouter` / `aya`), `--reasoning-effort` (o3, gpt-5, claude-opus-4-6), `--temperature`, `--concurrency`, `--openrouter-ignore-providers`, `--input-file` (local metadata JSON instead of `--split`), `--output-dir`. See `python api_code/main_api.py --help`.

### 4. Compare models

```bash
python src/result.py
```

Prints the extraction-rate and accuracy tables of all models in `results/` and saves them to `results/analysis_<split>_<prompt_language>.csv`.

---

## ✏️ Human-in-the-loop Enhancement

We provide web-based tools for:

* **Question Selection**: `src/choose_question.html` - Filter and select questions
* **OCR Verification**: `src/ocr_ground_truth.html` - Edit OCR results and descriptions
* **Quality Control**: `src/check_question.html` - Manual verification interface

---


## 🗂️ Repository Structure

```
VMMU/
├── api_code/                           # VLM evaluation
│   ├── api_handlers/                   # API wrapper for VLMs
│   └── main_api.py                     # Main API call logic
│
├── dataset/
│   ├── random_subset/                 # Sample exam questions by domain
│   ├── metadata/                      # Question annotations and ground truth
│   └── hf/                            # Images downloaded from Hugging Face (not tracked)
│
├── images/                            # Dataset overview images
├── vmmu_dataset.py                    # Loads a split from Hugging Face
├── requirements.txt
│
├── src/                               # Full pipeline for data extraction
│   ├── cut_question.py               # Question boundary detection
│   ├── convert_pdf_to_image.py       # PDF → PNG conversion
│   ├── check_question.html           # Manual verification interface
│   ├── choose_question.html          # Question selection tool
│   ├── ocr_ground_truth.html         # OCR verification tool
│   └── result.py                     # Accuracy analysis
│
└── api_key/                          # API credentials (or use environment variables)
    ├── claude_key.txt
    ├── openai_key.txt
    └── ...
```

---

## 📈 Key Findings

Our evaluation reveals several important insights:

1. **Thinking models perform best, but no model is reliable yet:** Open-source VLMs average only 37.35% accuracy and non-thinking closed-source VLMs 57.83%, both below the average Vietnamese high-school student (66.54%). Test-time compute via thinking models raises accuracy to 78.34% (best: Gemini-3-Pro, 86.33%), but this remains well below the expert reference (99.60%). IQ Test is consistently the hardest domain, while Geography and Math are comparatively easier
2. **OCR is a major bottleneck for open-source VLMs but not for closed-source ones:** Closed-source VLMs read embedded Vietnamese text reliably (mean BLEU 89.01%, CER 6.59%, WER 9.33%), while open-source VLMs score substantially lower (mean BLEU 57.98%). Open-source failures are therefore partly attributable to OCR, whereas closed-source failures point to multimodal grounding and reasoning
3. **Separating text from visual evidence improves reliability:** Giving the question and options as text alongside a crop of the visual evidence improves every evaluated VLM, by +7.97 points on average (+9.65 open-source, +8.56 non-thinking closed-source, +2.10 thinking closed-source), because models no longer have to read dense in-image text while grounding the visual evidence
4. **English translation does not help:** Translating both the text and the in-image text into English reduces accuracy by 2.47 points on average for closed-source VLMs, so native-language multimodal evaluation cannot be replaced by translation, even for models trained predominantly on English

---
