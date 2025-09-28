import torch

def sample(model, idx, max_new_tokens, context_length, temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        cond_idx = idx[:, -context_length:]

        with torch.no_grad():
            logits = model(cond_idx)
        # use only the last token's logits
        logits = logits[:, -1, :]
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits = torch.where(
                logits < v[:, -1], 
                torch.tensor(float('-inf')).to(logits.device), 
                logits
                )
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx