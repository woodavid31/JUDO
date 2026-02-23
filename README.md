## Introduction

This is the repository for ICLR 26 "JUDO: A Juxtaposed Domain-oriented Multimodal Reasoner for Industrial Anomaly QA"

## Installation

```bash
conda create -n judo python=3.10
conda activate judo
bash setup.sh
```

## Training

```bash
cd open-r1-multimodal
bash seg_sft_grpo.sh
```

## Dataset

You can download the dataset of MMAD from [here](https://huggingface.co/datasets/jiang-cc/MMAD)
You can download the dataset of REALIAD from [here](https://huggingface.co/datasets/Real-IAD/Real-IAD/tree/main/realiad_512)

## Evaluation

```bash
cd eval
python eval_seg_mult.py
```

## Acknowledgement

This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2025-02653113, High-Performance Research AI Computing Infrastructure Support at the 2 PFLOPS Scale)

