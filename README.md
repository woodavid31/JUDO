# JUDO (ICLR 2026)

Official implementation of  
[**"JUDO: A Juxtaposed Domain-oriented Multimodal Reasoner for Industrial Anomaly QA"**](https://openreview.net/forum?id=XW4mROtaVb&referrer=%5BAuthor+Console%5D%28%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions%29)

Base model: Qwen2.5-VL-7B  
Official trained checkpoint:  
https://huggingface.co/woodavid31/JUDO


## Installation

```bash
conda create -n judo python=3.10
conda activate judo
bash setup.sh
```

Multi-GPU is recommended for GRPO training.


## Training

The training script performs:

- Segmentation SFT  
- Domain knowledge SFT  
- GRPO alignment  

```bash
cd open-r1-multimodal
bash seg_sft_grpo.sh
```


## Dataset

Download datasets:

MMAD  
https://huggingface.co/datasets/jiang-cc/MMAD  

REAL-IAD  
https://huggingface.co/datasets/Real-IAD/Real-IAD/tree/main/realiad_512  

Place datasets according to the paths expected in the training scripts.


## Evaluation

```bash
cd eval
python eval_seg_mult.py
```

To evaluate the official JUDO model, set the model path to:

```
woodavid31/JUDO
```

Or replace it with your locally trained checkpoint.


## Output Format

Model outputs follow:

```
<seg>...</seg>
<think>...</think>
<answer>...</answer>
```



## Acknowledgement

This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2025-02653113, High-Performance Research AI Computing Infrastructure Support at the 2 PFLOPS Scale)
