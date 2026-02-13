import sys
sys.modules["flash_attn"] = None

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
import h5py
import argparse


def extract_and_save_llm_pairs_multilayer(
        first_model,
        second_model,
        save_path,
        layers,
        num_samples=2000,
        batch_size=8
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load sentences
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    sentences = [x['text'] for x in dataset if 25 < len(x['text']) < 500]
    print(f"[+] Loaded {len(sentences)} sentences from Wikitext-2.")
    print(f"[+] Extracting layers: {layers}")

    # Load both models
    tok_1 = AutoTokenizer.from_pretrained(first_model)
    tok_1.pad_token = tok_1.eos_token
    model_1 = AutoModelForCausalLM.from_pretrained(first_model, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager")

    tok_2 = AutoTokenizer.from_pretrained(second_model)
    tok_2.pad_token = tok_2.eos_token
    model_2 = AutoModelForCausalLM.from_pretrained(second_model, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager")

    model_1.eval()
    model_2.eval()

    # Pre-create HDF5
    with h5py.File(save_path, 'w') as f:
        f.attrs['layers'] = layers
        f.attrs['num_samples'] = len(sentences)

    first_batch = True
    dt = h5py.special_dtype(vlen=str)

    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i + batch_size]

            # Model 1
            inp_1 = tok_1(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            out_1 = model_1(**inp_1, output_hidden_states=True)
            seq_lens_1 = inp_1.attention_mask.sum(dim=1) - 1

            # Model 2
            inp_2 = tok_2(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            out_2 = model_2(**inp_2, output_hidden_states=True)
            seq_lens_2 = inp_2.attention_mask.sum(dim=1) - 1

            # Extract embeddings at each layer and save
            with h5py.File(save_path, 'a') as f:
                for layer_idx in layers:
                    emb_1 = out_1.hidden_states[layer_idx][torch.arange(len(batch)), seq_lens_1, :].cpu().numpy()
                    emb_2 = out_2.hidden_states[layer_idx][torch.arange(len(batch)), seq_lens_2, :].cpu().numpy()

                    key_1 = f"{first_model}/layer_{layer_idx}"
                    key_2 = f"{second_model}/layer_{layer_idx}"

                    if first_batch:
                        f.create_dataset(key_1, data=emb_1, maxshape=(None, emb_1.shape[1]))
                        f.create_dataset(key_2, data=emb_2, maxshape=(None, emb_2.shape[1]))
                    else:
                        f[key_1].resize(f[key_1].shape[0] + emb_1.shape[0], axis=0)
                        f[key_1][-emb_1.shape[0]:] = emb_1
                        f[key_2].resize(f[key_2].shape[0] + emb_2.shape[0], axis=0)
                        f[key_2][-emb_2.shape[0]:] = emb_2

                if first_batch:
                    f.create_dataset('sentences', data=batch, maxshape=(None,), dtype=dt)
                    first_batch = False
                else:
                    f['sentences'].resize(f['sentences'].shape[0] + len(batch), axis=0)
                    f['sentences'][-len(batch):] = batch

    del model_1, model_2, tok_1, tok_2
    torch.cuda.empty_cache()

    print(f"[+] Saved multi-layer embeddings to {save_path}")
    with h5py.File(save_path, 'r') as f:
        for layer_idx in layers:
            key_1 = f"{first_model}/layer_{layer_idx}"
            print(f"    {key_1}: {f[key_1].shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract multi-layer embeddings")
    parser.add_argument("--first_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--second_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--layers", type=str, default="8,16,24",
                        help="Comma-separated layer indices (e.g., 8,16,24)")
    parser.add_argument("--output", type=str, default="llama2_llama3_multilayer.h5")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    extract_and_save_llm_pairs_multilayer(
        args.first_model,
        args.second_model,
        save_path=args.output,
        layers=layers,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
