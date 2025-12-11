import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import time

MODEL_NAME = "microsoft/phi-2"
PROMPT = "In the distant future, humanity discovered"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K_VALUES = [5, 20, 50]
TOP_P_VALUES = [0.3, 0.7, 0.9]
COMBINATIONS = [
    {"k": 50, "p": 0.9},
    {"k": 10, "p": 0.5},
    {"k": 100, "p": 0.95}
]

def load_model():
    print(f"Device: {DEVICE}")
    print(f"Loading {MODEL_NAME}...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dtype_frac = torch.float16 if DEVICE == "cuda" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype_frac,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    
    if DEVICE == "cpu":
        model.to("cpu")
        
    print(f"Model loaded in {time.time() - start_time:.2f}s")
    return model, tokenizer

def run_generation(model, tokenizer, prompt, top_k, top_p, temperature=1.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.shape[1]
    
    set_seed(42)
    
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    elapsed = time.time() - t0
    
    return generated_text, elapsed

def main():
    model, tokenizer = load_model()
    
    print(f"\nPrompt: {PROMPT}")
    
    print("\n" + "="*30)
    print(" Top-k values")
    print(" (Top-p fixed at 1.0, Temp=1.0)")
    print("="*30)
    
    for k in TOP_K_VALUES:
        # top_p=1.0 отключает Nucleus sampling, чтобы мы видели чистый эффект Top-k
        result, t = run_generation(model, tokenizer, PROMPT, top_k=k, top_p=1.0)
        print(f"\n[Top-k: {k}]")
        print(f"Result: {result}")
        print(f"Time: {t:.2f}s")

    # --- 2. Исследование Top-p (Nucleus Sampling) ---
    print("\n" + "="*30)
    print(" Top-p values")
    print(" (Top-k disabled (set to 0), Temp=1.0)")
    print("="*30)
    
    for p in TOP_P_VALUES:
        # top_k=0 отключает Top-k выборку, работает только Top-p
        result, t = run_generation(model, tokenizer, PROMPT, top_k=0, top_p=p)
        print(f"\n[Top-p: {p}]")
        print(f"Result: {result}")
        print(f"Time: {t:.2f}s")

    # --- 3. Комбинации параметров ---
    print("\n" + "="*30)
    print(" Combinations")
    print("="*30)
    
    for combo in COMBINATIONS:
        k = combo["k"]
        p = combo["p"]
        # Здесь используем оба параметра вместе
        result, t = run_generation(model, tokenizer, PROMPT, top_k=k, top_p=p)
        print(f"\n[Top-k: {k} | Top-p: {p}]")
        print(f"Result: {result}")
        print(f"Time: {t:.2f}s")

if __name__ == "__main__":
    main()
