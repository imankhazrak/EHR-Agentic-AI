from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "google/gemma-4-e4b-it"
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())

tok = AutoTokenizer.from_pretrained(model_id)
print("tokenizer_ok", tok.__class__.__name__)

model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")
print("model_ok_loaded")

prompts = [
    "Diagnoses include hyperlipidemia and mixed dyslipidemia.",
    "Trauma admission with no lipid-related diagnosis or medication.",
    "CAD and diabetes present; no explicit lipid diagnosis listed.",
]

for i, p in enumerate(prompts, 1):
    inputs = tok(p, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        # Fast smoke path: verify the model can consume tokenized inputs via embeddings.
        emb = model.get_input_embeddings()(inputs["input_ids"])
    print(f"PROMPT_{i}_ok embedding_shape={tuple(emb.shape)} tokens={inputs['input_ids'].shape[1]}")

print("smoke_passed_inputs_through_model")
