def load_model(
    model_name: str,
    revision: str = "main",
    device: str = "cpu",
    dtype: str = "float32",
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA model requested but CUDA is unavailable")
    if dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("unsupported model dtype")
    torch_dtype = None if dtype == "auto" else getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch_dtype,
    )
    model.to(torch.device(device))
    model.eval()
    return tokenizer, model
