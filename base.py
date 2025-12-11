import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import time

# --- 1. Настройка устройства ---
# Проверяем, есть ли NVIDIA GPU. Если нет - используем CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Используется устройство: {device}")

# --- 2. Загрузка модели Phi-2 ---
# Phi-2 — это ~2.7 млрд параметров.
model_name = "microsoft/phi-2"

print(f"--> Начинаю загрузку модели {model_name}...")
start_time = time.time()

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Устанавливаем паддинг-токен (важно для Phi-2)
tokenizer.pad_token = tokenizer.eos_token

# Загружаем модель
# torch_dtype=torch.float16 уменьшает потребление памяти в 2 раза (важно для GPU)
# Если вы на CPU, float16 может не поддерживаться старыми процессорами, тогда используйте float32
dtype = torch.float16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True
)

if device == "cpu":
    model.to("cpu")

print(f"--> Модель загружена за {time.time() - start_time:.2f} секунд.")

# --- 3. Функция базовой генерации ---
def generate_text_basic(prompt):
    """
    Базовая функция генерации без сложных настроек сэмплирования.
    Использует жадный поиск (greedy search) по умолчанию, если не включить do_sample.
    """
    print(f"\n[PROMPT]: {prompt}")
    
    # Превращаем текст в цифры
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)
    
    # Генерация
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Ограничим длину для теста
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Декодируем цифры обратно в текст
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Выводим результат (убираем сам промпт для чистоты вывода, если нужно)
    result = text[len(prompt):].strip()
    print(f"[RESULT]: ...{result}")
    return result

# --- 4. Тестовый запуск ---
if __name__ == "__main__":
    test_prompt = "Large Language Models are"
    generate_text_basic(test_prompt)
    print("\n--> Этап 1 завершен успешно.")
