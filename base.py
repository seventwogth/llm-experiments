import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import time

# Пришлось сделать так, потому что на рабочем компьютере видеокарта не от NVIDIA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Ускоритель: {device}")

model_name = "microsoft/phi-2"

print(f"Модель: {model_name} загружается...")
start_time = time.time()

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Устанавливаем паддинг-токен (важно для Phi-2)
tokenizer.pad_token = tokenizer.eos_token

# Чтобы уменьшить потребление памяти моделью в 2 раза пробуем использовать float16
dtype_frac = torch.float16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=dtype_frac,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True
)

if device == "cpu":
    model.to("cpu")

print(f"Модель загружена за {time.time() - start_time:.2f} секунд.")

def generate_text_basic(prompt):

    print(f"\n[PROMPT]: {prompt}")
    
    # Превращаем текст в цифры
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Генерация
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Ограничим длину для теста
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Декодируем цифры обратно в текст
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Выводим результат
    result = text[len(prompt):].strip()
    print(f"[RESULT]:\n{result}")
    return result

# Промпты на английском, потому что модель не так хорошо работает с промптами на русском
if __name__ == "__main__":
    test_prompt = "В каком году была основана Священная Римская Империя?"
    generate_text_basic(test_prompt)
