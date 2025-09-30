import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer

adapter_path = None
model_path = "openai/gpt-oss-20b"  # 20B model - uncomment this line and comment the line above

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
messages = [{"role": "user", "content": "Giải thích thuật toán flash attention và cách tính toán của nó."}]


chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)
if adapter_path:
    model.load_adapter(adapter_path, adapter_name="multilingual-reasoner")
    model.set_adapter("multilingual-reasoner")
    print(f"Loaded adapter {adapter_path}")

model.eval()

# Tokenize and generate
inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
model.generate(
    **inputs,
    streamer=TextStreamer(tokenizer),
    max_new_tokens=1024,
)
