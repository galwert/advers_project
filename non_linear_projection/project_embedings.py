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

    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    sentences = [x['text'] for x in dataset if 50 < len(x['text']) < 200][:num_samples]

    # Extract first
    tok_1 = AutoTokenizer.from_pretrained(first_model)
    model_1 = AutoModelForCausalLM.from_pretrained(first_model, torch_dtype=torch.float16, device_map="auto")
    emb_1 = extract_hidden_states(model_1, tok_1, sentences, layer_idx, batch_size, device)
    del model_1
    torch.cuda.empty_cache()

    # Extract second
    tok_2 = AutoTokenizer.from_pretrained(second_model)
    model_2 = AutoModelForCausalLM.from_pretrained(second_model, torch_dtype=torch.float16, device_map="auto")
    emb_2 = extract_hidden_states(model_2, tok_2, sentences, layer_idx, batch_size, device)
    del model_2
    torch.cuda.empty_cache()

    with h5py.File(save_path, 'w') as f:
        f.create_dataset(first_model, data=emb_1.numpy())
        f.create_dataset(second_model, data=emb_2.numpy())
        f.attrs['layer_idx'] = layer_idx
        f.attrs['num_samples'] = len(sentences)