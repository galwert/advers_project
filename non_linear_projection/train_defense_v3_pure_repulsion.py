"""
Defense Training V3 - Pure Anchor Repulsion (Option 3)

NO harmful prompts at all - not even for computing refusal direction.
Just push away from anchor on benign data with very low coherency constraint.

This is the "cleanest" approach - tests if just being different from anchor
(in the right way with low coherency) is enough for defense.

Usage:
    python train_defense_v3_pure_repulsion.py \
        --defended_model meta-llama/Llama-2-7b-chat-hf \
        --anchor_model meta-llama/Meta-Llama-3-8B-Instruct \
        --projector_path projector.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import numpy as np
import gc
import os
import random
import warnings
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import List
from datasets import load_dataset
import argparse

from model import EmbeddingProjector

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.modules["flash_attn"] = None


@dataclass
class DefenseConfigV3Pure:
    """Configuration for pure repulsion defense training"""

    # Models
    defended_model_id: str = "meta-llama/Llama-2-7b-chat-hf"
    anchor_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    defended_type: str = "llama2"
    anchor_type: str = "llama3"

    # Projector
    projector_path: str = "projector.pt"
    projector_layer: int = 16

    # Training
    train_steps: int = 500
    lr: float = 5e-5
    batch_size: int = 4
    grad_accum: int = 2
    warmup_steps: int = 50

    # Loss weights - ONLY anchor repulsion + minimal coherency
    alpha: float = 1.0       # Anchor repulsion loss (main loss)
    beta: float = 0.01       # Coherency loss (keep VERY LOW)
    temperature: float = 0.07

    # Data
    n_benign_samples: int = 1000
    max_seq_length: int = 128

    # LoRA - larger capacity
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Output
    output_dir: str = "./defense_outputs_v3_pure"
    save_every: int = 100


BENIGN_PROMPTS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Write a Python function to calculate factorial.",
    "What are the primary colors?",
    "How do computers store data?",
    "Describe the water cycle.",
    "What is the speed of light?",
    "Explain the concept of gravity.",
    "What causes seasons on Earth?",
    "How does the internet work?",
    "What is machine learning?",
    "Describe the structure of DNA.",
    "What is the Pythagorean theorem?",
    "How do vaccines work?",
    "What is climate change?",
    "Tell me about the solar system.",
    "How do airplanes fly?",
    "What is electricity?",
    "Explain evolution.",
    "How does the heart work?",
]


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )


def format_prompt(text: str, model_type: str) -> str:
    model_type = model_type.lower()
    if model_type == "llama2":
        return f"[INST] {text} [/INST]"
    elif model_type == "llama3":
        return f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return text


def load_benign_prompts(n_samples: int = 1000) -> List[str]:
    prompts = list(BENIGN_PROMPTS)
    needed = n_samples - len(prompts)

    if needed <= 0:
        return prompts[:n_samples]

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for item in dataset:
            text = item['text'].strip()
            if 30 < len(text) < 300 and text[0].isalpha() and not text.startswith('='):
                clean = text.replace('\n', ' ').strip()[:150]
                if len(clean) > 20:
                    prompts.append(f"Tell me about: {clean}")
                if len(prompts) >= n_samples:
                    break
    except Exception as e:
        print(f"[!] Warning loading WikiText: {e}")
        while len(prompts) < n_samples:
            prompts.extend(BENIGN_PROMPTS)

    return prompts[:n_samples]


def anchor_repulsion_loss(h_defended_proj: torch.Tensor, h_anchor: torch.Tensor) -> torch.Tensor:
    """
    Push defended representations away from anchor in projected space.

    Loss = (1 + cos_sim) / 2
    This maps cosine similarity from [-1, 1] to [0, 1]
    Minimizing this pushes cos_sim toward -1 (maximum repulsion)
    """
    h_def_norm = F.normalize(h_defended_proj, dim=-1)
    h_anc_norm = F.normalize(h_anchor, dim=-1)
    cos_sim = F.cosine_similarity(h_def_norm, h_anc_norm, dim=-1)
    return (1 + cos_sim).mean() / 2


def train_defense_v3_pure(config: DefenseConfigV3Pure, device: str = "cuda") -> str:
    """
    Train defense using PURE anchor repulsion.
    NO harmful prompts, NO refusal direction - just push away from anchor.
    """

    print("\n" + "=" * 70)
    print("DEFENSE TRAINING V3 - PURE ANCHOR REPULSION")
    print("=" * 70)
    print(f"Defended: {config.defended_model_id}")
    print(f"Anchor: {config.anchor_model_id}")
    print(f"Alpha (anchor repulsion): {config.alpha}")
    print(f"Beta (coherency): {config.beta}")
    print(f"NO harmful prompts, NO refusal direction")
    print("=" * 70)

    os.makedirs(config.output_dir, exist_ok=True)

    # Load projector
    print(f"\n[*] Loading projector from {config.projector_path}")
    checkpoint = torch.load(config.projector_path, map_location=device, weights_only=False)
    d_defended = checkpoint['net.0.weight'].shape[1]
    d_anchor = checkpoint['net.2.weight'].shape[0]
    d_hidden = checkpoint['net.0.weight'].shape[0]

    projector = EmbeddingProjector(d_defended, d_anchor, d_hidden)
    projector.load_state_dict(checkpoint)
    projector.to(device)
    projector.eval()
    for p in projector.parameters():
        p.requires_grad = False

    # Load defended model
    print(f"\n[*] Loading defended model: {config.defended_model_id}")
    defended_tokenizer = AutoTokenizer.from_pretrained(config.defended_model_id, trust_remote_code=True)
    if not defended_tokenizer.pad_token:
        defended_tokenizer.pad_token = defended_tokenizer.eos_token
    defended_tokenizer.padding_side = "left"

    defended_model = AutoModelForCausalLM.from_pretrained(
        config.defended_model_id,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    defended_model = prepare_model_for_kbit_training(defended_model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM
    )
    defended_model = get_peft_model(defended_model, lora_config)
    defended_model.print_trainable_parameters()

    # Load anchor model
    print(f"\n[*] Loading anchor model: {config.anchor_model_id}")
    anchor_tokenizer = AutoTokenizer.from_pretrained(config.anchor_model_id, trust_remote_code=True)
    if not anchor_tokenizer.pad_token:
        anchor_tokenizer.pad_token = anchor_tokenizer.eos_token
    anchor_tokenizer.padding_side = "left"

    anchor_model = AutoModelForCausalLM.from_pretrained(
        config.anchor_model_id,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    anchor_model.eval()
    for p in anchor_model.parameters():
        p.requires_grad = False

    target_layer = config.projector_layer

    # Load ONLY benign prompts
    benign_prompts = load_benign_prompts(n_samples=config.n_benign_samples)
    print(f"\n[*] Training on {len(benign_prompts)} BENIGN prompts only")
    print(f"[*] NO harmful prompts used at all")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in defended_model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=0.01
    )

    # Training
    print(f"\n[*] Starting training for {config.train_steps} steps...")
    defended_model.train()
    metrics = {'total': [], 'repulsion': [], 'coherency': []}

    pbar = tqdm(range(config.train_steps), desc="Training")
    accumulated_loss = 0.0

    for step in pbar:
        # Warmup
        if step < config.warmup_steps:
            lr_scale = (step + 1) / config.warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = config.lr * lr_scale

        # Sample BENIGN batch only
        batch_prompts = random.sample(benign_prompts, min(config.batch_size, len(benign_prompts)))

        loss_repulsion_total = torch.tensor(0.0, device=device)
        loss_coherency_total = torch.tensor(0.0, device=device)

        for prompt in batch_prompts:
            defended_formatted = format_prompt(prompt, config.defended_type)
            anchor_formatted = format_prompt(prompt, config.anchor_type)

            def_enc = defended_tokenizer(
                defended_formatted, return_tensors="pt",
                truncation=True, max_length=config.max_seq_length
            ).to(device)

            anc_enc = anchor_tokenizer(
                anchor_formatted, return_tensors="pt",
                truncation=True, max_length=config.max_seq_length
            ).to(device)

            # Forward passes
            with torch.no_grad():
                anchor_out = anchor_model(**anc_enc, output_hidden_states=True)
                h_anchor = anchor_out.hidden_states[target_layer + 1][:, -1, :].float()

            defended_out = defended_model(**def_enc, output_hidden_states=True)
            h_defended = defended_out.hidden_states[target_layer + 1][:, -1, :].float()

            with defended_model.disable_adapter():
                with torch.no_grad():
                    base_out = defended_model(**def_enc, output_hidden_states=True)
                    h_base = base_out.hidden_states[target_layer + 1][:, -1, :].float()

            # 1. Anchor repulsion (MAIN loss)
            h_defended_proj = projector(h_defended)
            loss_rep = anchor_repulsion_loss(h_defended_proj, h_anchor)
            loss_repulsion_total = loss_repulsion_total + loss_rep

            # 2. Coherency loss (very weak - allow model to change)
            loss_coh = F.mse_loss(h_defended, h_base)
            loss_coherency_total = loss_coherency_total + loss_coh

        # Average
        n = len(batch_prompts)
        loss_repulsion_total = loss_repulsion_total / n
        loss_coherency_total = loss_coherency_total / n

        # Total loss - just repulsion + minimal coherency
        loss = config.alpha * loss_repulsion_total + config.beta * loss_coherency_total

        loss = loss / config.grad_accum
        loss.backward()
        accumulated_loss += loss.item()

        metrics['total'].append(loss.item() * config.grad_accum)
        metrics['repulsion'].append(loss_repulsion_total.item())
        metrics['coherency'].append(loss_coherency_total.item())

        if (step + 1) % config.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in defended_model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix({
                'loss': f"{accumulated_loss:.4f}",
                'rep': f"{np.mean(metrics['repulsion'][-10:]):.3f}",
                'coh': f"{np.mean(metrics['coherency'][-10:]):.4f}",
            })
            accumulated_loss = 0.0

        if (step + 1) % config.save_every == 0:
            ckpt = os.path.join(config.output_dir, f"checkpoint_{step+1}")
            defended_model.save_pretrained(ckpt)

    # Save final
    adapter_path = os.path.join(config.output_dir, "defense_adapter_final")
    defended_model.save_pretrained(adapter_path)
    print(f"\n[+] Final adapter: {adapter_path}")

    import json
    with open(os.path.join(config.output_dir, "metrics.json"), 'w') as f:
        json.dump({
            'config': {
                'alpha': config.alpha,
                'beta': config.beta,
                'train_steps': config.train_steps,
                'training_data': 'BENIGN_ONLY',
                'method': 'PURE_ANCHOR_REPULSION',
            },
            'final': {
                'repulsion': np.mean(metrics['repulsion'][-50:]),
                'coherency': np.mean(metrics['coherency'][-50:]),
            }
        }, f, indent=2)

    del defended_model, anchor_model, projector
    cleanup()

    return adapter_path


def main():
    parser = argparse.ArgumentParser(description="Defense Training V3 - Pure Anchor Repulsion")

    parser.add_argument("--defended_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--anchor_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--defended_type", type=str, default="llama2")
    parser.add_argument("--anchor_type", type=str, default="llama3")
    parser.add_argument("--projector_path", type=str, default="projector.pt")
    parser.add_argument("--projector_layer", type=int, default=16)

    parser.add_argument("--alpha", type=float, default=1.0, help="Anchor repulsion weight")
    parser.add_argument("--beta", type=float, default=0.01, help="Coherency weight (keep VERY LOW)")
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--train_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_benign_samples", type=int, default=1000)

    parser.add_argument("--output_dir", type=str, default="./defense_outputs_v3_pure")

    args = parser.parse_args()

    config = DefenseConfigV3Pure(
        defended_model_id=args.defended_model,
        anchor_model_id=args.anchor_model,
        defended_type=args.defended_type,
        anchor_type=args.anchor_type,
        projector_path=args.projector_path,
        projector_layer=args.projector_layer,
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temperature,
        train_steps=args.train_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        n_benign_samples=args.n_benign_samples,
        output_dir=args.output_dir,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_path = train_defense_v3_pure(config, device)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - PURE ANCHOR REPULSION")
    print("=" * 70)
    print(f"Adapter: {adapter_path}")
    print("\nEvaluate with:")
    print(f"  python evaluate_defense.py --skip_training --adapter_path {adapter_path}")


if __name__ == "__main__":
    main()
