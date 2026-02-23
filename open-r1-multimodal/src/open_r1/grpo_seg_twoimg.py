import sys, os
import numpy as np
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from math import exp

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
from math_verify import parse, verify

from open_r1.trainer import Qwen2VLGRPOTrainer_DIFF, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math
from deepspeed.runtime.zero import GatheredParameters
from sentence_transformers import SentenceTransformer
# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple

_semantic_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cuda"        
)

_semantic_model.eval()
for p in _semantic_model.parameters():
    p.requires_grad = False

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward
_SCRIPT_ARGS = None
_SYSTEM_PROMPT = None 
# ============================================================
# SYSTEM PROMPT FACTORY (uses actual grid size; rows/cols start at 1)
# ============================================================
def make_system_prompt(patch_rows: int, patch_cols: int) -> str:
    return (
        "You are a helpful assistant for industrial quality control tasks. The user asks a question, and the Assistant solves it.\n"
        f"The query image is divided into a grid of {patch_rows}×{patch_cols} patches. "
        "Patch coordinates are (row, col) and both row and col start from 1 (top‑left patch is (1,1)).\n\n"
        "Always respond in this exact order and format:\n"
        f"1) <seg> The anomalous region patch list (use (r,c) and (r,c_start)-(r,c_end) for contiguous columns). If there is not anomaly, respond 'None'. </seg>\n"
        "2) <think> Provide your detailed reasoning so that it leads to the correct answer. </think>\n"
        "3) <answer>A|B|C|D</answer>\n"
    )

# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format","structure","seg_f1","domain"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    grid_size: int = 16
    coverage_thresh: float = 0.02


_ANSWER_TAG = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
_THINK_TAG  = re.compile(r'<think>(.*?)</think>', re.DOTALL)
_SEG_TAG    = re.compile(r'<seg>(.*?)</seg>', re.DOTALL)


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        print("Randomly shuffled ======================")
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
   
        ex = self.list_data_dict[i]
        root = self.script_args.image_root

        prompt = [
                   {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": f"\nFirst image = QUERY; Second image = NORMAL template.\nQuestion: {ex['problem']}"},
                        ],
                    },
                ]


        def _load_img(rel):
            if not rel: return None
            path = os.path.join(root, rel)
            if os.path.exists(path):
                return Image.open(path).convert("RGB").resize((512,512), Image.LANCZOS)
            return None

        image = _load_img(ex.get("image"))
        template = _load_img(ex.get("template_image"))

        mask_path = None
        if ex.get("mask_path"):
            mp = os.path.join(root, ex["mask_path"])
            mask_path = mp if os.path.exists(mp) else None

        return {
            'image': image,
            'template_image': template,
            'mask_path': mask_path,
            'problem': ex['problem'],
            'solution': ex['solution'],
            'prompt': prompt,
            'domain_gt': ex['domain_gt']
        }


def extract_choice(text):
    if not text:
        return None

    text = re.sub(r'\s+', ' ', text) 
    
    if len(text) < 3:
        text = re.sub(r'[^\w\s]', '', text)      
        return text[0] if text else None

    choices = re.findall(r'(?<![A-Z])([A-D])(?![A-Z])', text)
    choices = re.findall(r'(?<![a-z])([A-D])(?![a-z])', text)

    if not choices:
        return None

    if len(choices) == 1:
        return choices[0]

    choice_scores = {choice: 0 for choice in choices}
    keywords = [
        'answer', 'correct', 'choose', 'select', 'right','think', 'believe', 'should'
    ]

    for choice in choices:
        pos = text.rfind(choice)
        context = text[max(0, pos-20):min(len(text), pos+20)]


        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        if pos > len(text) * 0.7:  
            choice_scores[choice] += 2

        if pos < len(text) - 1 and text[pos+1] in '。.!！,，':
            choice_scores[choice] += 1

    return max(choice_scores.items(), key=lambda x: x[1])[0]

_SEG_TAG = re.compile(r"<seg>(.*?)</seg>")

def _parse_grid_seg(txt: str):

    grid_size = _SCRIPT_ARGS.grid_size

    m = _SEG_TAG.search(txt or "")
    if not m:
        return set()

    payload = m.group(1).strip()
    if payload.lower() == 'none':
        return set()
    tokens = re.split(r",(?![^\(]*\))", payload)
    patches = set()

    try:
        for token in tokens:
            token = token.strip()
            if not token: continue # Skip empty strings from trailing commas

            if "-" in token:
                start_str, end_str = token.split("-")
                start_r, start_c = map(int, start_str.strip("() ").split(","))
                end_r, end_c = map(int, end_str.strip("() ").split(","))

                # --- VALIDATION ---
                is_valid = (
                    start_r == end_r and start_c <= end_c and
                    1 <= start_r <= grid_size and
                    1 <= start_c <= grid_size and
                    1 <= end_c <= grid_size
                )
                if not is_valid:
                    return -1 # Error: Invalid range or out of bounds

                for c in range(start_c, end_c + 1):
                    patches.add((start_r, c))
            else:
                r, c = map(int, token.strip("() ").split(","))

                # --- VALIDATION ---
                if not (1 <= r <= grid_size and 1 <= c <= grid_size):
                    return -1 # Error: Coordinate out of bounds
                patches.add((r, c))

    except (ValueError, IndexError):
        # This will catch the error from your traceback, where it tried
        # to convert '357) none' to an integer.
        return -1 # Error: Malformed string

    return patches

def _grid_from_mask_bool(mask_bool: np.ndarray, n: int, thr: float):
    h,w = mask_bool.shape
    rows = np.linspace(0,h,n+1).astype(int); cols = np.linspace(0,w,n+1).astype(int)
    pos=set()
    for r in range(n):
        for c in range(n):
            patch = mask_bool[rows[r]:rows[r+1], cols[c]:cols[c+1]]
            if patch.size and patch.mean()>=thr: pos.add((r+1,c+1))
    return pos
    

def structure_reward(completions, solution, **kwargs):

    contents = [
        comp[-1]["content"] if isinstance(comp, list) and comp else
        (comp[0]["content"] if isinstance(comp, list) and comp else "")
        for comp in completions
    ]
    
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    think_tag_pattern = r'<think>(.*?)</think>'
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        thinking = None
        answer_choice = None
        thinking_choice = None
        is_lazy_think = False
        early_answer_penalty = False
        
        try:
            # Extract the last <think> and <answer> content
            content_think_matches = re.findall(think_tag_pattern, content, re.DOTALL)
            content_answer_matches = re.findall(answer_tag_pattern, content, re.DOTALL)

            if content_think_matches:
                thinking = content_think_matches[-1].strip()
            
            if content_answer_matches:
                answer_text = content_answer_matches[-1].strip()
                answer_choice = extract_choice(answer_text)

            # ★ 1. Check for lazy think (starts with A/B/C/D)
            if thinking and re.match(r"^\s*[A-D][\s.:]", thinking):
                is_lazy_think = True
                reward = 0.0  # Hard zero if too lazy
            else:
                # Extract choice from thinking text for alignment check
                if thinking:
                    thinking_choice = extract_choice(thinking)

                # Original reward logic
                if thinking_choice is None and answer_choice is None:
                    reward = 0.0
                elif thinking_choice is None and answer_choice == sol:
                    reward = 0.8
                elif thinking_choice is None and answer_choice != sol:
                    reward = 0.05
                elif thinking_choice == sol and answer_choice == sol:
                    reward = 1.0
                elif thinking_choice != sol and answer_choice != sol and thinking_choice == answer_choice:
                    reward = 0.1
                else:
                    reward = 0.0

                # ★ 2. Check for "early answer mention" in first half of <think>
                if thinking:
                    print(thinking)
                    mid = len(thinking) // 2
                    first_half = thinking[:mid]
                    if re.search(r"\b([A-D])\b|([A-D])[\s.:]", first_half):
                        early_answer_penalty = True
                        reward = min(reward, 0.2)
        
        except Exception as e:
            reward = 0.0
            if os.getenv("DEBUG_MODE") == "true":
                print(f"Error processing content: {e}")

        # Debug logging
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "grpo_debug.log")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Choice reward: {reward} -------------\n")
                if is_lazy_think:
                    f.write("!!! LAZY THINK PENALTY APPLIED !!!\n")
                if early_answer_penalty:
                    f.write("!!! EARLY ANSWER PENALTY APPLIED !!!\n")
                f.write(f"Content: {content}\n")
                f.write(f"Thinking Text: {thinking}\n")
                f.write(f"Extracted Answer Choice: {answer_choice}\n")
                f.write(f"Extracted Thinking Choice: {thinking_choice}\n")
                f.write(f"Correct Solution: {sol}\n")
        
        rewards.append(reward)
        
    return rewards

def choice_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion has the correct answer (A, B, C, or D)
    and if the thinking process supports the correct answer."""
    contents = [completion[-1]["content"] if isinstance(completion, list) else completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        try:
            content_answer_matches = re.findall(answer_tag_pattern, content, re.DOTALL)
            if content_answer_matches:
                content_answer = content_answer_matches[-1].strip()
                answer_choice = extract_choice(content_answer)
            else:
                content_answer = None
                answer_choice = None
                
            if answer_choice == sol:
                reward = 1.0
            else:
                reward = 0.0
                answer_choice = None
                content_answer_matches = None

        except Exception as e:
            content_answer_matches = f"Answer extract error: {str(e)}"
        
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Choice easy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Extracted answer: {answer_choice}\n")
                f.write(f"GPT Answer: {content_answer_matches}\n")
                f.write(f"Correct Answer: {sol}\n")
        
        rewards.append(reward)
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has the correct format with <think> and <answer> tags."""
    # Extract content properly from different completion structures
    contents = [completion[-1]["content"] if isinstance(completion, list) else completion[0]["content"] for completion in completions]
    
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    # Pattern to check for both <think> and <answer> tags in proper order
    pattern = r'<seg>.*?</seg>.*?<think>.*?</think>.*?<answer>.*?</answer>'
    #pattern = r'<eval>.*?</eval>.*?<think>.*?</think>.*?<answer>.*?</answer>'
    
    for content in contents:
        # Check if the basic format pattern exists in the content
        reward = 0
        if re.search(pattern, content, re.DOTALL):
            reward = 0.2
        else:
            reward = 0.0
        
        rewards.append(reward)
        
        # Add debugging output if needed
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                has_think = "<think>" in content and "</think>" in content
                has_answer = "<answer>" in content and "</answer>" in content
                f.write(f"Has think tags: {has_think}, Has answer tags: {has_answer}\n")
    
    return rewards





def extract_options(text):
    """
    "A: ... B: ... C: ... D: ..." 형태에서
    {'A': "...", 'B': "...", 'C': "...", 'D': "..."} dict 추출
    """
    pattern = r"([A-D]):\s*(.*?)(?=(?:[A-D]:|$))"
    matches = re.findall(pattern, text, re.S)
    return {m[0]: m[1].strip() for m in matches}



def _dist(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def segmentation_grid_f1_reward(completions, **kwargs):
    out = []
    N   = _SCRIPT_ARGS.grid_size
    THR = _SCRIPT_ARGS.coverage_thresh
    alpha = getattr(_SCRIPT_ARGS, "soft_alpha", 1.0) 

    mps = kwargs.get("mask_path", [None]*len(completions))

    for comp, mp in zip(completions, mps):
        txt = comp[-1]["content"] if isinstance(comp, list) else comp[0]["content"]
        pred = _parse_grid_seg(txt)  
        if pred == -1:
            out.append(0.0) 
            continue
        if not mp or not os.path.exists(mp):
            if not pred:
                out.append(1.0)
            else:
                out.append(0.0)
            continue

        gt = Image.open(mp).convert("L").resize((512,512), Image.NEAREST)
        gt = np.array(gt) > 0
        gt_cells = _grid_from_mask_bool(gt, N, THR)  
        if not pred and not gt_cells:
            out.append(1.0)  
            continue

        if not pred or not gt_cells:
            out.append(0.0) 
            continue

        soft_prec = 0.0
        for p in pred:
            best = max(exp(-alpha * _dist(p,g)) for g in gt_cells)
            soft_prec += best
        soft_prec /= len(pred)

        soft_rec = 0.0
        for g in gt_cells:
            best = max(exp(-alpha * _dist(g,p)) for p in pred)
            soft_rec += best
        soft_rec /= len(gt_cells)

        if soft_prec + soft_rec > 0:
            f1 = 2 * soft_prec * soft_rec / (soft_prec + soft_rec)
        else:
            f1 = 0.0

        reward = 0.2 + 0.8*f1
        out.append(float(reward))

    return out


def segmentation_grid_real_f1_reward(completions, **kwargs):

    out = []
    N   = _SCRIPT_ARGS.grid_size
    THR = _SCRIPT_ARGS.coverage_thresh
    # The 'alpha' parameter is no longer needed for the hard F1 score.

    mps = kwargs.get("mask_path", [None]*len(completions))

    for comp, mp in zip(completions, mps):
        txt = comp[-1]["content"] if isinstance(comp, list) else comp[0]["content"]
        pred = _parse_grid_seg(txt)
        if pred == -1:
            out.append(0.0)
            continue

        if not mp or not os.path.exists(mp):
            if not pred:
                out.append(1.0)
            else:
                out.append(0.0)
            continue

        gt = Image.open(mp).convert("L").resize((512,512), Image.NEAREST)
        gt = np.array(gt) > 0
        gt_cells = _grid_from_mask_bool(gt, N, THR)

        if not pred and not gt_cells:
            out.append(1.0)
            continue
        

        if not pred or not gt_cells:
            out.append(0.0)
            continue


        pred_set = set(pred)
        gt_set = set(gt_cells)


        tp = len(pred_set.intersection(gt_set))
        fp = len(pred_set.difference(gt_set))
        fn = len(gt_set.difference(pred_set))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        

        reward = 0.2 + 0.8 * f1
        out.append(float(reward))

    return out
            
def domain_reward(completions, **kwargs):
    contents = [completion[-1]["content"] if isinstance(completion, list) else completion[0]["content"] for completion in completions]
    rewards = []
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    think_tag_pattern = r'<think>(.*?)</think>'

    gts = kwargs["domain_gt"]

    for content, gt in zip(contents,gts):
        reward = 0.0
        
        # First try to find if there's a proper format with <think> and <answer> tags
        match = re.search(r'<seg>(.*?)</seg>\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>', content, re.DOTALL)

        if match:
            content_answer_matches = re.findall(answer_tag_pattern, content, re.DOTALL)
            content_think_matches = re.findall(think_tag_pattern, content, re.DOTALL)
            
            
            if content_think_matches:
                content_think = content_think_matches[-1].strip()

                with torch.no_grad():
                    embedding1 = _semantic_model.encode([content_think],convert_to_numpy=True, convert_to_tensor=False, normalize_embeddings=True)[0]
                    embedding2 = _semantic_model.encode([gt],convert_to_numpy=True, convert_to_tensor=False, normalize_embeddings=True)[0]
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    
                    
                reward = similarity
        else:
            reward = 0.0

        rewards.append(reward*0.1)
    return rewards

reward_funcs_registry = {
                        "accuracy": choice_reward,
                        "format": format_reward,
                        "structure": structure_reward,
                        "seg_f1": segmentation_grid_real_f1_reward,
                        "domain" : domain_reward
                        }


def main(script_args, training_args, model_args):
    global _SCRIPT_ARGS, _SYSTEM_PROMPT
    _SCRIPT_ARGS = script_args
    _SYSTEM_PROMPT = make_system_prompt(script_args.grid_size, script_args.grid_size)

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)
    
    trainer_cls = Qwen2VLGRPOTrainer_DIFF
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__=="__main__":
    parser=TrlParser((GRPOScriptArguments,GRPOConfig,ModelConfig))
    script_args,training_args,model_args=parser.parse_args_and_config()
    main(script_args,training_args,model_args)
