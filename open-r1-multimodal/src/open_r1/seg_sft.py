#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import io, os, sys, json, math, random, logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import datasets
import transformers
from transformers import (
    AutoProcessor, AutoTokenizer, set_seed,
    Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
)
from transformers.utils import is_flash_attn_2_available
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    ModelConfig, ScriptArguments, SFTTrainer, SFTConfig, TrlParser,
    get_kbit_device_map, get_peft_config, get_quantization_config,
)

import yaml
from transformers import TrainerCallback, TrainerState, TrainerControl

logger = logging.getLogger(__name__)
processor = None  # set later


# ---------------------------
# Script arguments
# ---------------------------
@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: Optional[str] = field(default=None, metadata={"help": "Root directory of images/masks"})
    grid_size: int = field(default=8, metadata={"help": "N for N×N grid"})
    coverage_thresh: float = field(default=0.05, metadata={"help": "Patch is anomalous if >= this fraction"})
    # ★ NEW: default diff text when query is normal and gt is null/empty
    default_normal_diff: str = field(
        default="No meaningful difference is observed; the query matches the normal template in alignment, shape, and surface condition.",
        metadata={"help": "Fallback <diff> text for normal queries when gt is null."}
    )


# ---------------------------
# Utilities
# ---------------------------
def _load_image(path: str) -> Image.Image:
    """Loads and resizes an image to 512x512."""
    with open(path, "rb") as f:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    # ★ MODIFIED: Resize the image to a fixed 512x512 resolution
    return img.resize((512, 512), Image.LANCZOS)

def _load_mask(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        m = Image.open(io.BytesIO(f.read()))
    if m.mode != "L":
        m = m.convert("L")
    arr = np.array(m)
    return arr > 0  # boolean mask

def _grid_from_mask(mask: np.ndarray, n: int, thr: float) -> List[Tuple[int, int]]:
    h, w = mask.shape
    rows = np.linspace(0, h, n + 1).astype(int)
    cols = np.linspace(0, w, n + 1).astype(int)
    hits = []
    for r in range(n):
        for c in range(n):
            patch = mask[rows[r]:rows[r+1], cols[c]:cols[c+1]]
            if patch.size and (patch.mean() >= thr):
                hits.append((r + 1, c + 1))  # 1-based
    if not hits:
        for r in range(n):
            for c in range(n):
                patch = mask[rows[r]:rows[r+1], cols[c]:cols[c+1]]
                if patch.size and (patch.mean() > 0):
                    hits.append((r + 1, c + 1))  # 1-based
    return hits

def _format_answer(coords: List[Tuple[int, int]]) -> str:
    """
    Produce "The anomalous area is ..." with rowwise run-length compression.
    """
    if not coords:
        return "None"

    by_row: Dict[int, List[int]] = {}
    for r, c in coords:
        by_row.setdefault(r, set()).add(c)  # dedupe
    parts: List[str] = []

    for r in sorted(by_row):
        cols = sorted(by_row[r])
        start = prev = cols[0]
        for x in cols[1:]:
            if x == prev + 1:
                prev = x
            else:
                if start == prev:
                    parts.append(f"({r},{start})")
                else:
                    parts.append(f"({r},{start})-({r},{prev})")
                start = prev = x
        # flush last run
        if start == prev:
            parts.append(f"({r},{start})")
        else:
            parts.append(f"({r},{start})-({r},{prev})")

    return " , ".join(parts)

# ★ NEW: build the two-line supervised target with tags
def _build_tagged_target(diff_text: str, seg_text: str) -> str:
    diff_text = diff_text.strip()
    seg_text = seg_text.strip()
    return f"<seg> {seg_text} </seg>\n<think> {diff_text} </think>"

# ★ UPDATED: prompt now demands TWO tagged lines, exactly.
def _build_messages(q_img: Image.Image, n_img: Image.Image, n_grid: int, target_text: str) -> List[Dict[str, Any]]:
    # ★ NEW: system prompt
    system_text = (
        f"You are a helpful assistant for industrial quality control tasks. The user asks a question, and the Assistant solves it.\n"
        f"Divide the query image into a {n_grid}x{n_grid} grid numbered (row,col) from (1,1) at top-left.\n\n"
        f"Always respond in this exact order and format:\n"
        f"  1) <seg> The anomalous region patch list (use (r,c) and (r,c_start)-(r,c_end) for contiguous columns). </seg>\n"
        f"  2) <think> A concise comparison describing differences between query and normal. </think>\n"
        f"If there is no anomaly, the <seg> line must be: 'None'.\n"
    )
    user_text = (
         f"First image = QUERY; second image = NORMAL template.\n"
         f"Describe the anomalous region in the query image and give your comparative reason for it."
    )

    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "image", "image": q_img},
            {"type": "image", "image": n_img},
            {"type": "text", "text": user_text},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
    ]

def _norm_path(root: Optional[str], maybe_path) -> Optional[str]:
    if maybe_path is None:
        return None
    p = str(maybe_path).strip()
    if p == "" or p.lower() in {"none", "null"}:
        return None
    if os.path.isabs(p) or (root is None):
        return p
    return os.path.join(root, p)


# ---------------------------
# Dataset: YAML -> JSONL entries -> multi-image messages
# ---------------------------
class AnomalyMultiImageDataset(Dataset):
    def __init__(self, data_yaml: str, script_args: SFTScriptArguments):
        self.root = script_args.image_root
        self.N = script_args.grid_size
        self.T = script_args.coverage_thresh
        self.default_normal_diff = script_args.default_normal_diff  # ★ NEW
        self.samples: List[Dict[str, Any]] = []

        if not data_yaml.endswith(".yaml"):
            raise ValueError("Provide a YAML that lists datasets with json_path entries.")

        with open(data_yaml, "r") as f:
            y = yaml.safe_load(f)
        blocks = y.get("datasets", [])
        for blk in blocks:
            json_path = blk.get("json_path")
            sampling = blk.get("sampling_strategy", "all")
            num = None
            if ":" in sampling:
                sampling, num = sampling.split(":")

            # load file
            if json_path.endswith(".jsonl"):
                with open(json_path, "r") as jf:
                    cur = [json.loads(line) for line in jf if line.strip()]
            elif json_path.endswith(".json"):
                cur = json.load(open(json_path, "r"))
            else:
                raise ValueError(f"Unsupported file: {json_path}")

            # sampling
            if num is not None:
                k = math.ceil(int(num.rstrip("%")) * 0.01 * len(cur)) if "%" in num else int(num)
                if sampling == "first": cur = cur[:k]
                elif sampling == "end": cur = cur[-k:]
                elif sampling == "random":
                    random.shuffle(cur); cur = cur[:k]

            kept = []
            for ex in cur:
                q = _norm_path(self.root, ex.get("image_path"))
                n = _norm_path(self.root, ex.get("normal_image_path"))
                m = _norm_path(self.root, ex.get("mask_path"))  # may be None
                gt = ex.get("gt", None)  # ★ NEW

                if not q or not n:
                    continue
                if not (os.path.exists(q) and os.path.exists(n)):
                    continue
                if m is not None and not os.path.exists(m):
                    continue

                kept.append({"q": q, "n": n, "m": m, "gt": gt})  # ★ include gt
            logger.info(f"Loaded {len(kept)} samples from {json_path}")
            self.samples.extend(kept)

        if not self.samples:
            raise RuntimeError("No valid samples found. Check paths in YAML and image_root.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        rec = self.samples[i]
        q_img = _load_image(rec["q"])
        n_img = _load_image(rec["n"])

        # seg line
        if rec["m"] is None:
            coords = []  # no anomaly
        else:
            mask = _load_mask(rec["m"])
            coords = _grid_from_mask(mask, self.N, self.T)
        seg_text = _format_answer(coords)

        # diff line (gt or default)
        gt_raw = rec.get("gt")
        if gt_raw is None or (isinstance(gt_raw, str) and gt_raw.strip().lower() in {"", "none", "null"}):
            # Use default iff NO anomaly (coords==[])
            diff_text = self.default_normal_diff if not coords else "A localized difference exists in the regions indicated by the segmentation line."
        else:
            diff_text = str(gt_raw)

        target = _build_tagged_target(diff_text, seg_text)  # ★ two lines with tags
        msgs = _build_messages(q_img, n_img, self.N, target)
        return {"messages": msgs}


# ---------------------------
# Collator: multi-image + left padding + prompt-only labels
# ---------------------------
def _extract_images(messages: List[Dict[str, Any]]) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list): content = [content]
        for el in content:
            if isinstance(el, dict) and el.get("type") == "image" and isinstance(el.get("image"), Image.Image):
                imgs.append(el["image"].convert("RGB"))
    return imgs

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # full (user+assistant) texts and image lists
    full_texts = [
        processor.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False, add_vision_id=True
        ).strip()
        for ex in examples
    ]
    full_images = [_extract_images(ex["messages"]) for ex in examples]

    # prompt-only (exclude assistant) → lengths
    prompt_texts = [
        processor.apply_chat_template(
            ex["messages"][:-1], tokenize=False, add_generation_prompt=True, add_vision_id=True
        ).strip()
        for ex in examples
    ]
    prompt_images = [_extract_images(ex["messages"][:-1]) for ex in examples]

    prompt_lens: List[int] = []
    for t, imgs in zip(prompt_texts, prompt_images):
        pi = processor(text=t, images=imgs, padding=False, return_tensors="pt")
        prompt_lens.append(int(pi["input_ids"].shape[1]))

    # encode full batch (LEFT padding)
    batch = processor(text=full_texts, images=full_images, padding=True, return_tensors="pt")

    labels = batch["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    # mask the leftmost non-pad tokens equal to prompt_len per sample
    attn = batch["attention_mask"]
    for i, L in enumerate(prompt_lens):
        nonpad_idx = torch.nonzero(attn[i].bool(), as_tuple=False).squeeze(-1)
        if L <= len(nonpad_idx):
            labels[i, nonpad_idx[:L]] = -100

    # mask vision tokens too (conservative)
    if hasattr(processor, "image_token"):
        img_tok_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        if img_tok_id is not None and img_tok_id != -1:
            labels[labels == img_tok_id] = -100

    batch["labels"] = labels
    return batch


# ---------------------------
# Simple logger (optional)
# ---------------------------
class SimpleGenerationLogger(TrainerCallback):
    def __init__(self, processor, dataset, max_new_tokens=120, log_frequency=10):
        self.processor = processor
        self.dataset = dataset
        self.max_new_tokens = max_new_tokens
        self.log_frequency = log_frequency

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.log_frequency != 0:
            return
        model = kwargs.get("model")
        if model is None: return
        try:
            idx = random.randint(0, len(self.dataset) - 1)
            ex = self.dataset[idx]
            prompt_text = self.processor.apply_chat_template(
                ex["messages"][:-1], tokenize=False, add_generation_prompt=True, add_vision_id=True
            )
            imgs = _extract_images(ex["messages"])
            model.eval()
            with torch.no_grad():
                inputs = self.processor(text=[prompt_text], images=[imgs], return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                out_ids = model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            inp_len = inputs["input_ids"].shape[1]
            gen_text = self.processor.tokenizer.decode(out_ids[0][inp_len:], skip_special_tokens=True).strip()
            print("\n" + "="*80)
            print(f" Step {state.global_step} (sample #{idx})")
            print("-"*40)
            print("GT:", ex["messages"][-1]["content"][0]["text"])
            print("-"*40)
            print("GEN:", gen_text)
            print("="*80 + "\n", flush=True)
        except Exception as e:
            print(f"[Logger ERROR @ step {state.global_step}]: {e}", flush=True)
        finally:
            model.train()


# ---------------------------
# Main
# ---------------------------
def main(script_args, training_args, model_args):
    # Repro + logging
    set_seed(training_args.seed)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed: {bool(training_args.local_rank != -1)}, fp16: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Resume if needed
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected at {last_checkpoint}.")

    # Dataset
    dataset = AnomalyMultiImageDataset(script_args.dataset_name, script_args)

    # Processor (force LEFT padding)
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor (vision-language).")
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor (vision-language).")

    # Model init
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quant_cfg = get_quantization_config(model_args)
    attn_impl = model_args.attn_implementation
    if attn_impl == "flash_attention_2" and not is_flash_attn_2_available():
        logger.warning("flash-attn 2 not found; falling back to sdpa")
        attn_impl = "sdpa"

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=attn_impl,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quant_cfg is not None else None,
        quantization_config=quant_cfg,
    )

    if "Qwen2-VL" in model_args.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Trainer config
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False

    callbacks = [SimpleGenerationLogger(processor, dataset, max_new_tokens=120, log_frequency=1000)]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=callbacks,
    )

    # Train
    logger.info("*** Train ***")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if getattr(training_args, "push_to_hub", False):
        try:
            logger.info("Pushing to hub...")
            trainer.push_to_hub()
        except Exception as e:
            logger.warning(f"push_to_hub failed (skipping): {e}")

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)