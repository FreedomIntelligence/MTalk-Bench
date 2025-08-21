# MTalk-Bench: Evaluating Speech-to-Speech Models in Multi-Turn Dialogues via Arena-style and Rubrics Protocols

**MTalk-Bench** is a benchmark designed to evaluate **speech-to-speech (S2S) large language models (LLMs)** in **realistic, multi-turn dialogue scenarios**. It offers both **arena-style** and **rubric-based** evaluation protocols to comprehensively assess models across a diverse range of **linguistic**, **paralinguistic**, and **acoustic** dimensions.

## 🗒 MTalk-Bench Overview

The benchmark covers the following core evaluation aspects:
- **Semantic Information**: Understanding & Memory, Reasoning & Execution, Interaction Strategy, Security Assessment, Pragmatic & Culture.
- **Paralinguistic Information**: Paralinguistic Comprehension, Paralinguistic Generation.
- **Ambient Sound**: Ambient Sound Perception, Multiparty Interaction.

MTalk-Bench is designed to reflect real-world conversational challenges and support fair, transparent, and extensible evaluation of next-generation S2S models.


## 📁 Repository Structure
```text
MTalk-Bench/
├── data/
├── src/                     # Source codes for Audio LLM automated evaluation
│   ├── audio_arena_style.py
│   └── audio_rubric_based.py
└── asset
```

## 🗃️ Dataset Access

The MTalk-Bench dataset (including audio files, transcribed texts, and testing prompts) is available on [🤗 MTalk-Bench](https://huggingface.co/datasets/FreedomIntelligence/MTalk-Bench) under a research license.


## 🚀 Quick Start

Follow the steps below to get started with **MTalk-Bench** evaluation.


### 1. Clone the Repository
```bash
git clone https://github.com/FreedomIntelligence/MTalk-Bench.git
cd MTalk-Bench
```

### 2. Download the Dataset
```bash
huggingface-cli download \
    --repo-type dataset \
    --resume-download \
    ./FreedomIntelligence/MTalk-Bench \
    --local-dir MTalk-Bench-Data
```
This will download the complete dataset (audio, transcripts, prompts) into MTalk-Bench-Data/.


### 3. Prepare Your Model Output
Run your chosen **speech-to-speech (S2S)** model on the MTalk-Bench dataset and generate **audio** responses.  
Format your results as a `.json` file according to the required schema (example in `./data/sample.json`) and place it in the `data/` directory.

### 4. Run Evaluation
You can choose between **arena-style** and **rubric-based** evaluations, and select the type of information to evaluate (`semantic`, `paralinguistic`, or `ambient`).
```bash
# Arena-style example:
python ./src/audio_arena_api.py \
    --eval_type semantic \
    --judge_model gpt-4o-audio-preview \
    --new_data_file ./data/sample.json

# Rubric-based example:

python ./src/audio_rubric_api.py \
    --eval_type paralinguistic \
    --judge_model gemini-2.5-pro \
    --new_data_file ./data/sample.json
```
Available parameters:
- eval_type: `semantic`, `paralinguistic`, `ambient`
- judge_model: `gpt-4o-audio-preview`, `gemini-2.5-pro`
- new_data_file: path to your `.json` result file



## 📄 Citation
If you use MTalk-Bench in your research, please cite:

```bibtex
@misc{mtalkbench2025,
  title={MTalk-Bench: Evaluating Speech-to-Speech Models in Multi-Turn Dialogues via Arena-style and Rubrics Protocols},
  author={Yuhao Du, Qianwei Huang, Guo Zhu, Zhanchen Dai, Sunian Chen, Qiming Zhu, Yuhao Zhang, Li Zhou, and Benyou Wang},
  year={2025},
  url={[xxxxxxxx](https://freedomintelligence.github.io/MTalk-Bench/)},
}
```
