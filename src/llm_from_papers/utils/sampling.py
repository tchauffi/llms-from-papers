import torch
import tiktoken


def get_tokens_from_text(tokenizer: tiktoken.core.Encoding, text: str) -> torch.Tensor:
    """Convert text to tokens and return as a tensor.

    Args:
        tokenizer: tiktoken tokenizer instance.
        text: Input text string.

        Returns:
        Tensor of token IDs.
    """
    tokens = tokenizer.encode(text)
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # add batch dimension


def get_text_from_tokens(
    tokenizer: tiktoken.core.Encoding, tokens: torch.Tensor
) -> str:
    """Convert a tensor of token IDs back to text.
    Args:
        tokenizer: tiktoken tokenizer instance.
        tokens: Tensor of token IDs.
    Returns:
        Decoded text string.
    """
    tokens = tokens.squeeze(0).tolist()  # remove batch dimension
    text = tokenizer.decode(tokens)
    return text


def sample(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_k: int = None,
    eos_id: int = None,
) -> torch.Tensor:
    """Generate text from a model using sampling with temperature and top-k filtering.
    Args:
        model: The language model to use for generation.
        idx: Tensor of shape (1, sequence_length) containing the initial token IDs.
        max_new_tokens: Maximum number of new tokens to generate.
        context_length: The maximum context length the model can handle.
        temperature: Sampling temperature. Higher values increase randomness.
        top_k: If specified, only consider the top_k tokens with highest probability.
        eos_id: If specified, stop generation when this token ID is generated.
    Returns:
        Tensor of shape (1, sequence_length + max_new_tokens) containing the generated token IDs.
    """
    for _ in range(max_new_tokens):
        cond_idx = idx[:, -context_length:]

        with torch.no_grad():
            logits = model(cond_idx)
        # use only the last token's logits
        logits = logits[:, -1, :]
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits = torch.where(
                logits < v[:, -1], torch.tensor(float("-inf")).to(logits.device), logits
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
