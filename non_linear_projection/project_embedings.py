from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
import h5py

def extract_hidden_states(model, tokenizer, sentences, layer_idx, batch_size=8, device='cuda'):
    model.eval()
    embeddings = []

    tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs, output_hidden_states=True)

            # Get last non-padded token for each sequence
            seq_lens = inputs.attention_mask.sum(dim=1) - 1
            hidden = outputs.hidden_states[layer_idx]
            batch_emb = hidden[torch.arange(len(batch)), seq_lens, :]

            embeddings.append(batch_emb.cpu())

    return torch.cat(embeddings, dim=0)

def extract_and_save_llm_pairs(
        first_model,
        second_model,
        save_path,
        num_samples=2000,
        layer_idx=16,
        batch_size=8
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load sentences
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    sentences = [x['text'] for x in dataset if 50 < len(x['text']) < 200][:num_samples]

    # Load both models
    tok_1 = AutoTokenizer.from_pretrained(first_model)
    tok_1.pad_token = tok_1.eos_token
    model_1 = AutoModelForCausalLM.from_pretrained(first_model, torch_dtype=torch.float16, device_map="auto")

    tok_2 = AutoTokenizer.from_pretrained(second_model)
    tok_2.pad_token = tok_2.eos_token
    model_2 = AutoModelForCausalLM.from_pretrained(second_model, torch_dtype=torch.float16, device_map="auto")

    model_1.eval()
    model_2.eval()

    # Pre-create HDF5
    with h5py.File(save_path, 'w') as f:
        f.attrs['layer_idx'] = layer_idx
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
            emb_1 = out_1.hidden_states[layer_idx][torch.arange(len(batch)), seq_lens_1, :].cpu().numpy()

            # Model 2 (same batch)
            inp_2 = tok_2(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            out_2 = model_2(**inp_2, output_hidden_states=True)
            seq_lens_2 = inp_2.attention_mask.sum(dim=1) - 1
            emb_2 = out_2.hidden_states[layer_idx][torch.arange(len(batch)), seq_lens_2, :].cpu().numpy()

            # Save
            with h5py.File(save_path, 'a') as f:
                if first_batch:
                    f.create_dataset(first_model, data=emb_1, maxshape=(None, emb_1.shape[1]))
                    f.create_dataset(second_model, data=emb_2, maxshape=(None, emb_2.shape[1]))
                    f.create_dataset('sentences', data=batch, maxshape=(None,), dtype=dt)
                    first_batch = False
                else:
                    f[first_model].resize(f[first_model].shape[0] + emb_1.shape[0], axis=0)
                    f[first_model][-emb_1.shape[0]:] = emb_1
                    f[second_model].resize(f[second_model].shape[0] + emb_2.shape[0], axis=0)
                    f[second_model][-emb_2.shape[0]:] = emb_2
                    f['sentences'].resize(f['sentences'].shape[0] + len(batch), axis=0)
                    f['sentences'][-len(batch):] = batch

    del model_1, model_2, tok_1, tok_2
    torch.cuda.empty_cache()

def main():
    # --- CONFIG ---
    model_a = "meta-llama/Llama-2-7b-chat-hf"
    model_b = "mistralai/Mistral-7B-Instruct-v0.1"
    extract_and_save_llm_pairs(model_a, model_b, save_path="")


if __name__ == '__main__':
    main()