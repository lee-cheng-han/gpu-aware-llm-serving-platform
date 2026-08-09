def load_model(model_name: str, revision: str = "main", device: str = "cpu"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    model.to(torch.device(device))
    model.eval()
    return tokenizer, model
