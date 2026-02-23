# Copyright 2025
# License: Apache-2.0

import logging
import os
import sys
import math
import json
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

import datasets
import transformers
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    set_seed,
)
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from configs import SFTConfig  # your existing config
from utils.callbacks import get_callbacks  # your existing callback factory

import yaml

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------

@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: Optional[str] = field(
        default=None, metadata={"help": "Root directory of the images"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "YAML file that points to JSON/JSONL datasets"}
    )
    sft_system_prompt: str = field(
        default="You are a helpful assistant for industrial quality control tasks.",
        metadata={"help": "Stable system identity/policy. Keep identical for SFT and GRPO."},
    )
    task_instruction_text: str = field(
        default=(
            "You are given a representative non-defect image and a domain-specific question. "
            "Answer the question clearly and accurately without interpreting or describing the image."
        ),
        metadata={"help": "Task-specific framing to place in the USER message."},
    )
    # Optional: resize safeguard (kept conservative)
    max_width: int = field(default=1000)
    max_height: int = field(default=800)


# --------------------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------------------
processor = None  # set in main()


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

class LazySupervisedDataset(Dataset):
    """
    Reads a YAML manifest:
      datasets:
        - json_path: /path/to/file.jsonl
          sampling_strategy: "all" | "first:N" | "end:N" | "random:N" | "first:10%" ...
    JSON/JSONL entries must include:
      - image (relative path under image_root)
      - problem
      - solution
    """

    def __init__(self, data_path: str, script_args: SFTScriptArguments):
        super().__init__()
        self.args = script_args
        self.items: List[Dict[str, Any]] = []

        if not data_path.endswith(".yaml"):
            raise ValueError(f"Unsupported file type: {data_path} (expecting .yaml)")

        with open(data_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        datasets_ = yaml_data.get("datasets", [])
        for spec in datasets_:
            json_path = spec.get("json_path")
            sampling_strategy = spec.get("sampling_strategy", "all")
            if json_path.endswith(".jsonl"):
                cur = []
                with open(json_path, "r") as jf:
                    for line in jf:
                        cur.append(json.loads(line.strip()))
            elif json_path.endswith(".json"):
                with open(json_path, "r") as jf:
                    cur = json.load(jf)
            else:
                raise ValueError(f"Unsupported file in YAML: {json_path}")

            # parse sampling strategy
            count = None
            if ":" in sampling_strategy:
                kind, n = sampling_strategy.split(":")
                if "%" in n:
                    p = int(n.replace("%", ""))
                    count = math.ceil(p * len(cur) / 100)
                else:
                    count = int(n)
                sampling_strategy = kind

            if sampling_strategy == "first" and count is not None:
                cur = cur[:count]
            elif sampling_strategy == "end" and count is not None:
                cur = cur[-count:]
            elif sampling_strategy == "random" and count is not None:
                random.shuffle(cur)
                cur = cur[:count]
            # "all" -> keep as is

            logger.info(f"Loaded {len(cur)} samples from {json_path}")
            self.items.extend(cur)

        self.image_root = self.args.image_root
        self.max_w = self.args.max_width
        self.max_h = self.args.max_height
        self.system_prompt = self.args.sft_system_prompt
        self.task_instruction = self.args.task_instruction_text

    def __len__(self):
        return len(self.items)

    def _load_image(self, rel_path: str) -> Optional[Image.Image]:
        if rel_path is None:
            return None
        full = os.path.join(self.image_root, rel_path)
        # if missing, sample another entry until found (same behavior as your version)
        tries = 0
        while not os.path.exists(full) and tries < 5:
            logger.warning(f"Image not found: {full}. Sampling a different example.")
            ex = random.choice(self.items)
            rel_path = ex.get("image", None)
            if rel_path is None:
                return None
            full = os.path.join(self.image_root, rel_path)
            tries += 1
        if not os.path.exists(full):
            logger.error(f"Image still not found after retries: {full}")
            return None

        img = Image.open(full).convert("RGB")
        img = img.resize((512,512), Image.LANCZOS)
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.items[idx]
        image = self._load_image(ex.get("image"))
        user_text = (
            f"{self.task_instruction}\n\n"
            f"Question: {ex['problem']}"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
            {"role": "assistant", "content": ex["solution"]},
        ]

        return {
            "image": image,
            "problem": ex["problem"],
            "solution": ex["solution"],
            "messages": messages,
        }



def _render_chat(messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    # Renders a single conversation to a string using the processor's chat template.
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

    # Render full texts
    full_texts = [_render_chat(ex["messages"], add_generation_prompt=False) for ex in examples]
    # Render prefixes: messages without final assistant, but with generation prompt (assistant header)
    prefix_texts = [
        _render_chat(ex["messages"][:-1], add_generation_prompt=True)
        for ex in examples
    ]
    # Images
    images = [ex["image"] for ex in examples]

    # Tokenize full batch (padded)
    batch = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Tokenize prefixes individually (no padding) to get true per-sample lengths with image tokens
    prefix_lens: List[int] = []
    for t, img in zip(prefix_texts, images):
        tok = processor(
            text=[t],
            images=[img] if img is not None else None,
            return_tensors="pt",
            padding=False,
        )
        prefix_lens.append(tok["input_ids"].shape[1])

    # Build labels from input_ids, mask everything except the assistant continuation
    labels = input_ids.clone()
    pad_id = processor.tokenizer.pad_token_id
    image_token_id = getattr(processor, "image_token", None)
    if image_token_id is not None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)

    for i in range(len(examples)):
        # mask prefix (system+user+assistant header)
        labels[i, :prefix_lens[i]] = -100
        # mask pads
        labels[i, input_ids[i] == pad_id] = -100
        # mask image tokens if present
        if image_token_id is not None:
            labels[i, input_ids[i] == image_token_id] = -100

    batch["labels"] = labels
    return batch


# --------------------------------------------------------------------------------------
# Live preview logger (unchanged, but ensures same system/user layout)
# --------------------------------------------------------------------------------------

class SimpleGenerationLogger(TrainerCallback):
    def __init__(self, processor, dataset: Dataset, max_new_tokens=150, log_frequency=50):
        self.processor = processor
        self.dataset = dataset
        self.max_new_tokens = max_new_tokens
        self.log_frequency = log_frequency

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == 0 or state.global_step % self.log_frequency != 0:
            return
        model = kwargs.get("model")
        if model is None:
            return
        try:
            sample_idx = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[sample_idx]

            # Render prefix (messages without assistant) with generation prompt
            input_text = processor.apply_chat_template(
                sample["messages"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=[input_text],
                images=[sample["image"]] if sample["image"] is not None else None,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            model.eval()
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            inp_len = inputs["input_ids"].shape[1]
            new_tokens = gen_ids[0][inp_len:]
            gen_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            print("\n" + "=" * 80)
            print(f"Step {state.global_step} | Sample #{sample_idx}")
            print("-" * 40)
            print("Question:")
            print(sample["problem"][:300])
            print("-" * 40)
            print("Ground truth:")
            print(sample["solution"][:300])
            print("-" * 40)
            print("Generation:")
            print(gen_text)
            print("=" * 80 + "\n", flush=True)
        except Exception as e:
            print(f"[Preview ERROR @ step {state.global_step}] {e}", flush=True)
        finally:
            model.train()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(script_args: SFTScriptArguments, training_args: SFTConfig, model_args: ModelConfig):
    set_seed(training_args.seed)

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(training_args.get_process_log_level())
    datasets.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed: {bool(training_args.local_rank != -1)}, fp16: {training_args.fp16}"
    )
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Script parameters: {script_args}")
    logger.info(f"Train args: {training_args}")

    # Resume
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint and training_args.resume_from_checkpoint is None:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    # Dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    # Processor
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

    # Ensure pad tokens defined
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Model
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    if "Qwen2-VL" in model_args.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)


    # Trainer
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False

    callbacks = get_callbacks(training_args, model_args)
    callbacks.append(SimpleGenerationLogger(processor, dataset, max_new_tokens=200, log_frequency=1000))

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