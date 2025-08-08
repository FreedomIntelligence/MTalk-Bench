# MTalk-Bench: Evaluating Speech-to-Speech Models in Multi-Turn Dialogues via Arena-style and Rubrics Protocols

**MTalk-Bench** is a benchmark designed to rigorously evaluate **speech-to-speech (S2S) large language models (LLMs)** in **realistic, multi-turn dialogue scenarios**. It offers both **arena-style** and **rubric-based** evaluation protocols to comprehensively assess models across a diverse range of **linguistic**, **paralinguistic**, and **acoustic** dimensions.

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
├── eval_script                # Dataset preparation scripts and metadata
├── eval_prompt/                   # Prompt used in evaluation
│   ├── audio_based/      
│   │   ├── arena/                 
│   │   └── rubrics/
│   └── transcribed_text_based/
│       ├── arena/
│       └── rubrics/
├── config.yaml             # Benchmark configuration
└── README.md               # You are here!

```

## 🗃️ Dataset Access

The MTalk-Bench dataset (including audio files and transcribed texts) is available on [🤗 MTalk-Bench](https://huggingface.co/datasets/FreedomIntelligence/MTalk-Bench) under a research license.


## 🚀 Quick Start

Follow the steps below to get started with **MTalk-Bench** evaluation.


### 1. Download the Dataset and Evaluation Prompts
- Download the **MTalk-Bench dataset** (including audio files and transcribed texts) from the [🤗 MTalk-Bench](https://huggingface.co/datasets/FreedomIntelligence/MTalk-Bench).
- Clone this GitHub repository to obtain the official evaluation code and prompts.

```bash
git clone https://github.com/your-org/MTalk-Bench.git
cd MTalk-Bench
```
> Make sure the downloaded dataset and the eval/ folder (containing prompts) are placed in the correct structure under MTalk-Bench/.



### 2. Choose a Model and Generate Audio Outputs   
- Select a **speech-to-speech (S2S)** model you want to evaluate. Run the model on the MTalk-Bench dataset to generate responses in audio format.
After inference, organize the model outputs following the required file format (see below), and save them in the `data/` folder.


### 3. Format Your Results

Your results should be stored in a `.jsonl` file, where each line represents one sample in the following format:

```json
{
TBD
}
```

- Place your `.jsonl` file under the `data/` directory.
- Example filename: `data/my_s2s_model.jsonl`

> ⚠️ Ensure all paths are relative to the root directory or absolute to avoid file not found errors.



### 4. Run Evaluation

Use the evaluation scripts under `eval_script/` to assess your model using either **arena-style** or **rubric-based** protocols.

```bash
# Placeholder: Scripts to be released soon
python eval_script/audio_based_eval_arena.py --input data/my_s2s_model.jsonl --config config.yaml
```



### 5. Analyze Results

The evaluation outputs (e.g., scores, rankings, annotations) will be saved to the `results/` directory by default. You can further process or visualize them based on your research needs.

#### Sample Output Format



## 📄 Citation
If you use MTalk-Bench in your research, please cite:

```bibtex
@misc{mtalkbench2025,
  title={MTalk-Bench: Evaluating Speech-to-Speech Models in Multi-Turn Dialogues via Arena-style and Rubrics Protocols},
  author={xxxxxx},
  year={2025},
  url={xxxxxxxx},
}
```
