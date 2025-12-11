import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import time

MODEL_NAME = "microsoft/phi-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRESETS = {
    "creative": {
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "desc": "Для более свободной генерации"
    },
    "precise": {
        "temperature": 0.1,
        "top_k": 10,
        "top_p": 0.5,
        "desc": "Для более логичной генерации"
    }
}

class LLMInteract:
    def __init__(self):
        print(f"Устройство: {DEVICE}")
        print(f"Загрузка модели {MODEL_NAME}...")
        t0 = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        dtype_frac = torch.float16 if DEVICE == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=dtype_frac,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True
        )
        
        if DEVICE == "cpu":
            self.model.to("cpu")
            

    def generate(self, prompt, settings, max_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_len = inputs.input_ids.shape[1]
        
        set_seed(42)
        
        print(f"\n[Генерация...] ({settings['desc']})")
        t0 = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=settings["temperature"],
                top_k=settings["top_k"],
                top_p=settings["top_p"],
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        elapsed = time.time() - t0
        
        return generated_text, elapsed

    def run_benchmark(self):
        tasks = [
            ("Открытое утверждение", "Artificial Intelligence will eventually", "creative"),
            ("Вопрос", "What are the three main states of matter?", "precise"),
            ("Творческое задание", "Write a short poem about a robot seeking love.", "creative"),
            ("Запрос на создание списка", "List 5 healthy fruits:", "precise")
        ]
        
        print("\n" + "="*40)
        print(" ЗАПУСК БЕНЧМАРКА (Типы промптов)")
        print("="*40)
        
        for task_name, prompt, mode in tasks:
            print(f"\n>>> Тип: {task_name}")
            print(f"Prompt: {prompt}")
            
            settings = PRESETS[mode]
            result, t = self.generate(prompt, settings)
            
            print(f"Result:\n{result.strip()}")
            print(f"Time: {t:.2f}s")
            print("-" * 20)

def main_loop():
    bot = LLMInteract()
    
    while True:
        print("\n" + "="*40)
        print("МЕНЮ:")
        print("1. Ввести свой промпт")
        print("2. Запустить бенчмарк (все типы промптов)")
        print("3. Выход")
        choice = input("Выбор: ")
        
        if choice == "1":
            prompt = input("\nВведите промпт (English): ")
            if not prompt: continue
            
            print("Выберите режим:")
            print("1 - Creative")
            print("2 - Precise")
            mode_in = input("Режим (Enter для Creative): ")
            
            mode_key = "precise" if mode_in == "2" else "creative"
            settings = PRESETS[mode_key]
            
            res, t = bot.generate(prompt, settings)
            print(f"\n[RESULT]: {res}")
            print(f"[TIME]: {t:.2f}s")
            
        elif choice == "2":
            bot.run_benchmark()
            
        elif choice == "3":
            print("Выход...")
            break
        else:
            print("Неверный ввод")

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nПринудительное завершение.")
