#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
import torch
from tqdm import tqdm
from PIL import Image
import sys
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
import time

# Append project root to path if needed
try:
    sys.path.append("..")
    from summary import caculate_accuracy_mmad
except ImportError:
    print("Warning: 'caculate_accuracy_mmad' from 'summary' module not found. Final accuracy calculation will be skipped.")
    def caculate_accuracy_mmad(filepath):
        print(f"Skipping accuracy calculation for {filepath}.")

# Import model-specific libraries
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

# Setup logging
log_dir = "inference_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"multi_image_eval_perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger("MultiImageThinkAD_Perf")

# System prompt is now generated dynamically
SYSTEM_PROMPT = None

def make_system_prompt(patch_rows: int, patch_cols: int) -> str:
    """Generates the system prompt for multi-image input."""
    return (
        "You are a helpful assistant for industrial quality control tasks. The user asks a question, and the Assistant solves it.\n"
        f"The query image is divided into a grid of {patch_rows}×{patch_cols} patches. "
        "Patch coordinates are (row, col) and both row and col start from 1 (top‑left patch is (1,1)).\n\n"
        "Always respond in this exact order and format:\n"
        f"1) <seg> The anomalous region patch list (use (r,c) and (r,c_start)-(r,c_end) for contiguous columns). If there is not anomaly, respond 'None'. </seg>\n"
        "2) <think> Provide your detailed reasoning so that it leads to the correct answer. </think>\n"
        "3) <answer>A|B|C|D</answer>\n"
    )

class OptimizedMultiGPUEngine:
    """Performance-optimized multi-GPU inference engine for multi-image input."""
    def __init__(self, model_path, num_gpus=None, enable_kv_cache=True, compile_model=False):
        self.model_path = model_path
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.enable_kv_cache, self.compile_model = enable_kv_cache, compile_model
        self.models, self.processors, self.tokenizers = {}, {}, {}
        self.gpu_locks = {i: threading.Lock() for i in range(self.num_gpus)}
        self.image_cache, self.cache_lock = {}, threading.Lock()
        logger.info(f"Initializing {self.num_gpus} GPU inference engines with FlashAttention.")
        self._load_models()

    def _load_models(self):
        for gpu_id in range(self.num_gpus):
            logger.info(f"Loading model on GPU {gpu_id}")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": f"cuda:{gpu_id}"},
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            ).eval()

            if self.compile_model and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info(f"Compiled model on GPU {gpu_id}")
                except Exception as e:
                    logger.warning(f"Compilation failed on GPU {gpu_id}: {e}")

            self.models[gpu_id] = model
            self.processors[gpu_id] = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.processors[gpu_id].tokenizer.padding_side = "left"
            self.tokenizers[gpu_id] = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def infer_batch(self, batch_data, sub_batch_size):
        gpu_batches = [[] for _ in range(self.num_gpus)]
        for idx, item in enumerate(batch_data):
            gpu_batches[idx % self.num_gpus].append(item)
        
        all_results = []
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = [executor.submit(self._process_gpu, gbatch, gid, sub_batch_size)
                       for gid, gbatch in enumerate(gpu_batches) if gbatch]
            for fn in as_completed(futures):
                try:
                    all_results.extend(fn.result())
                except Exception as e:
                    logger.error(f"Error in GPU batch: {e}")
        return all_results

    def _process_gpu(self, items, gpu_id, sub_batch_size):
        results = []
        for start in range(0, len(items), sub_batch_size):
            chunk = items[start:start+sub_batch_size]
            results.extend(self._run_chunk(chunk, gpu_id))
        return results

    def _run_chunk(self, chunk, gpu_id):
        with self.gpu_locks[gpu_id]:
            model, processor, tokenizer = self.models[gpu_id], self.processors[gpu_id], self.tokenizers[gpu_id]
            texts, imgs, metas = [], [], []

            for query_image, template_image, qdata, path, gt, qtype in chunk:
                sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
                usr_msg = {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": f"nFirst image = QUERY; Second image = NORMAL template.\nQuestion: {qdata['text']}\n{qdata['opts_text']}"}
                ]}
                chat = processor.apply_chat_template([sys_msg, usr_msg], tokenize=False, add_generation_prompt=True)
                texts.append(chat)
                imgs.extend([query_image, template_image])
                metas.append((path, qdata, gt, qtype))

            inputs = processor(text=texts, images=imgs, padding=True, return_tensors='pt').to(f"cuda:{gpu_id}")
            
            gen_kwargs = {
                "max_new_tokens": 1024,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "use_cache": self.enable_kv_cache
            }

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs).cpu()
            dt = (time.time() - t0) / len(chunk)

            res = []
            for o, meta in zip(out, metas):
                path, qdata, gt, qtype = meta
                decoded = tokenizer.decode(o[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                seg = re.search(r'<seg>(.*?)</seg>', decoded, re.DOTALL)
                think = re.search(r'<think>(.*?)</think>', decoded, re.DOTALL)
                ans = re.search(r'<answer>([A-D])</answer>', decoded)
                pred = ans.group(1) if ans else (re.search(r'\b([A-D])\b', decoded).group(1) if re.search(r'\b([A-D])\b', decoded) else 'X')

                res.append({
                    "image": path, "question": qdata['text'], "question_type": qtype,
                    "segmentation": seg.group(1).strip() if seg else '',
                    "thinking": think.group(1).strip() if think else '',
                    "gpt_answer": pred, "correct_answer": gt, "inference_time": dt, "gpu_id": gpu_id
                })

                right_text = "O"
                if gt != pred:
                    right_text = "X"
                print(f"GPU{gpu_id} {path} GT:{gt} PRED:{pred} -> {right_text}")
            
            del inputs, out
            torch.cuda.empty_cache()
            return res

class AsyncImageLoader:
    def __init__(self, max_workers=16):
        self.exec, self.cache = ThreadPoolExecutor(max_workers=max_workers), {}
    def load(self, img_path, data_root):
        def _ld():
            full_path = os.path.join(data_root, img_path)
            if full_path in self.cache: return self.cache[full_path]
            try:
                im = Image.open(full_path).convert('RGB').resize((512, 512), Image.LANCZOS)
                self.cache[full_path] = im
                return im
            except Exception as e:
                logger.error(f"Image load error {img_path}: {e}")
                return None
        return self.exec.submit(_ld)
    def shutdown(self): self.exec.shutdown()

def parse_conversation(text_gt):
    qs, ans, types = [], [], []
    for k, v in text_gt.items():
        if k.startswith('conversation'):
            for idx, qa in enumerate(v):
                items = list(qa['Options'].items())
                random.shuffle(items)
                txt, dict_opts, correct = '', {}, None
                for i, (opt, text) in enumerate(items):
                    label = chr(65 + i)
                    txt += f"{label}. {text}\n"
                    dict_opts[label] = text
                    if opt == qa['Answer']: correct = label
                q_data = {'text': qa['Question'], 'opts_text': txt, 'options': dict_opts, 'qid': idx}
                qs.append(q_data)
                ans.append(correct)
                types.append(qa.get('type', 'unknown'))
            break
    return qs, ans, types

def create_batches(data, bs):
    for i in range(0, len(data), bs): yield data[i:i+bs]

def main():
    parser = argparse.ArgumentParser(description="Run high-performance multi-image evaluation for VLMs.")
    parser.add_argument('--model_path',type=str,
                        default='Place the directory for the trained model')
    parser.add_argument('--save_name',type=str, default='JUDO')
    parser.add_argument('--data_path',type=str, default='JUDO/MMAD')
    parser.add_argument('--json_path',type=str, default='JUDO/MMAD/mmad.json')
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--sub_batch_size',type=int, default=16)
    parser.add_argument('--num_gpus',type=int, default=None)
    parser.add_argument('--grid_size', type=int, default=16, help="Grid size (N for N*N grid) to use in the system prompt.")
    parser.add_argument('--enable_kv_cache',action='store_true')
    parser.add_argument('--compile_model',action='store_true')
    args = parser.parse_args()

    global SYSTEM_PROMPT
    SYSTEM_PROMPT = make_system_prompt(args.grid_size, args.grid_size)
    logger.info(f"Using multi-image system prompt for a {args.grid_size}x{args.grid_size} grid.")

    with open(args.json_path, 'r', encoding='utf-8') as f: chat_ad = json.load(f)

    out_fp = f"result/answers_twoimg_think_{args.save_name}.json"
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    
    existing_results = []
    processed_items = set()
    if os.path.exists(out_fp):
        logger.info(f"Resuming from existing file: {out_fp}")
        with open(out_fp, 'r', encoding='utf-8') as f:
            content = f.read()
        for match in re.finditer(r'{\s*"image":.*?\s*}', content, re.DOTALL):
            try:
                item = json.loads(match.group(0))
                key = (item['image'], item['question'])
                if key not in processed_items:
                    existing_results.append(item)
                    processed_items.add(key)
            except json.JSONDecodeError:
                logger.warning("Found a malformed JSON object in the existing file, skipping it.")
        if existing_results:
            logger.info(f"Recovered {len(existing_results)} completed items.")

    engine = OptimizedMultiGPUEngine(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        enable_kv_cache=args.enable_kv_cache,
        compile_model=args.compile_model
    )
    
    loader = AsyncImageLoader()
    
    data_with_futures = []
    for query_path, meta in tqdm(chat_ad.items(), desc='Submitting image load tasks'):
        if 'random_templates' in meta and meta['random_templates']:
            template_path = random.choice(meta['random_templates'])
            query_future = loader.load(query_path, args.data_path)
            template_future = loader.load(template_path, args.data_path)
            data_with_futures.append((query_future, template_future, meta, query_path))
        else:
            logger.warning(f"Skipping {query_path} due to missing 'random_templates'.")

    all_data = []
    for q_future, t_future, meta, query_path in tqdm(data_with_futures, desc='Preparing data'):
        try:
            query_image = q_future.result()
            template_image = t_future.result()
            if query_image and template_image:
                qs, as_, ts = parse_conversation(meta)
                for q, gt, qt in zip(qs, as_, ts):
                    all_data.append((query_image, template_image, q, query_path, gt, qt))
        except Exception as e:
            logger.error(f'Error resolving futures for {query_path}: {e}')
    loader.shutdown()
    
    data_to_process = [
        item for item in all_data
        if (item[3], item[2]['text']) not in processed_items
    ]

    logger.info(f"Total items: {len(all_data)} | Processed: {len(processed_items)} | Remaining: {len(data_to_process)}")

    if not data_to_process:
        logger.info("All tasks have already been completed. No inference needed.")
        caculate_accuracy_mmad(out_fp)
        return

    total_start_time = time.time()
    with open(out_fp, 'w', encoding='utf-8') as f:
        f.write("[\n")
        
        is_first_entry = True
        if existing_results:
            for entry in existing_results:
                if not is_first_entry:
                    f.write(",\n")
                json.dump(entry, f, ensure_ascii=False, indent=2)
                is_first_entry = False
        
        for batch in tqdm(list(create_batches(data_to_process, args.batch_size)), 'Processing Remaining Batches'):
            results = engine.infer_batch(batch, args.sub_batch_size)
            for entry in results:
                if not is_first_entry:
                    f.write(",\n")
                json.dump(entry, f, ensure_ascii=False, indent=2)
                f.flush()
                is_first_entry = False
        
        f.write("\n]")

    total_duration = time.time() - total_start_time
    avg_time = total_duration / len(data_to_process) if data_to_process else 0
    logger.info(f'Total inference time for this run: {total_duration:.2f}s | Avg per new sample: {avg_time:.3f}s')
    logger.info(f'Results saved to {out_fp}')
    
    caculate_accuracy_mmad(out_fp)

if __name__ == '__main__':
    main()