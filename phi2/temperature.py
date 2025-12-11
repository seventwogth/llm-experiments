import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import time

# Параметры
MODEL_NAME = "microsoft/phi-2"
PROMPT = "The secret to a successful life is"
TEMPERATURES = [0.1, 0.4, 0.7, 1.0]

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dtype_frac = torch.float16 if device == "cuda" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype_frac,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    if device == "cpu":
        model.to("cpu")
        
    return model, tokenizer, device

def experiment_temperature(model, tokenizer, device, prompt, temps):
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    for temp in temps:
        set_seed(42) 
        
        print(f"\n Temperature: {temp}")
        t0 = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=temp,
                top_k=0,                 
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Декодируем только новые токены
        generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        print(f"Result:\n{generated_text}")
        print(f"Time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    model, tokenizer, device = load_model()
    experiment_temperature(model, tokenizer, device, PROMPT, TEMPERATURES)
