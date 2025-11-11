from PIL import Image
import torch
import einops  # 明示的にインポート（transformersのインポートチェック前に必要）
from transformers import AutoModelForCausalLM, AutoProcessor
import time

print(f"torch version: {torch.__version__}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 全体の処理時間計測開始
total_start_time = time.time()

CHOSEN_TASK = "ocr"  # Options: 'ocr' | 'table' | 'chart' | 'formula'
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

print(f"DEVICE: {DEVICE}")

print("Starting model loading...")
model_path = "PaddlePaddle/PaddleOCR-VL"
image_path = "test.png"

print(f"Model path: {model_path}")
print(f"Image path: {image_path}")

# 画像読み込み時間計測
image_load_start = time.time()
image = Image.open(image_path).convert("RGB")
image_load_time = time.time() - image_load_start
print(f"Image loaded in {image_load_time:.3f} seconds")

# モデルのロード（プログレスバーを表示）
print("Downloading/loading model (this may take several minutes on first run)...")
model_load_start = time.time()
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # メモリ使用量を削減
        device_map="auto" if DEVICE == "cuda" else None  # GPU使用時は自動デバイスマッピング
    )
    if DEVICE == "cpu":
        model = model.to(DEVICE)
    model = model.eval()
    model_load_time = time.time() - model_load_start
    print(f"Model loaded successfully in {model_load_time:.2f} seconds")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying to load without dtype specification...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto" if DEVICE == "cuda" else None
    )
    if DEVICE == "cpu":
        model = model.to(DEVICE)
    model = model.eval()
    model_load_time = time.time() - model_load_start
    print(f"Model loaded successfully (without bfloat16) in {model_load_time:.2f} seconds")

processor_load_start = time.time()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
processor_load_time = time.time() - processor_load_start
print(f"Processor loaded successfully in {processor_load_time:.3f} seconds")

messages = [
    {"role": "user",         
     "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPTS[CHOSEN_TASK]},
        ]
    }
]

print("Preparing inputs...")
input_prep_start = time.time()
inputs = processor.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 	
    return_dict=True,
    return_tensors="pt"
).to(DEVICE)
input_prep_time = time.time() - input_prep_start
print(f"Inputs prepared in {input_prep_time:.3f} seconds")

print("Generating output (this may take a while)...")
print("Note: Large models can take several minutes to generate output.")
print("If it takes too long, you can interrupt with Ctrl+C and try with smaller max_new_tokens.")

# 生成パラメータを調整（より高速化）
inference_start = time.time()

with torch.no_grad():  # メモリ使用量を削減
    try:
        # より軽量な設定で生成
        print("Starting generation with optimized parameters...")
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,  # 256に減らして高速化
            min_new_tokens=10,   # 最小トークン数
            do_sample=False,
            num_beams=1,  # ビームサーチを無効化して高速化
            pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, 'pad_token_id') else None,
            eos_token_id=processor.tokenizer.eos_token_id if hasattr(processor.tokenizer, 'eos_token_id') else None,
            use_cache=True,
            early_stopping=True  # 早期終了を有効化
        )
        inference_time = time.time() - inference_start
        print(f"Generation completed in {inference_time:.2f} seconds!")
    except Exception as e:
        print(f"Error during generation: {e}")
        print("Trying with even simpler parameters...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,  # さらに減らす
            do_sample=False,
            num_beams=1
        )
        inference_time = time.time() - inference_start
        print(f"Generation completed in {inference_time:.2f} seconds!")

# デコード時間計測
decode_start = time.time()
outputs = processor.batch_decode(outputs, skip_special_tokens=True)[0]
decode_time = time.time() - decode_start

# 全体の処理時間
total_time = time.time() - total_start_time

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(outputs)
print("="*60)

# 時間計測結果の表示
print("\n" + "="*60)
print("PERFORMANCE METRICS:")
print("="*60)
print(f"Image loading time:        {image_load_time:>8.3f} seconds")
print(f"Model loading time:        {model_load_time:>8.2f} seconds")
print(f"Processor loading time:   {processor_load_time:>8.3f} seconds")
print(f"Input preparation time:   {input_prep_time:>8.3f} seconds")
print(f"Inference time:           {inference_time:>8.2f} seconds")
print(f"Decoding time:            {decode_time:>8.3f} seconds")
print("-" * 60)
print(f"Total processing time:    {total_time:>8.2f} seconds")
print("="*60)

