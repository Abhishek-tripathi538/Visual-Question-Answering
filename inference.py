import os
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# =============== CONFIG =================
MODEL_PATH = "./blip2_lora_finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============== LOAD MODEL ===============
print("📦 Loading processor and BLIP2 model...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
base_model = Blip2ForConditionalGeneration.from_pretrained(
    "ybelkada/blip2-opt-2.7b-fp16-sharded",
    device_map="auto",
    quantization_config=quant_config,
    torch_dtype=torch.float16
)

# Load LoRA fine-tuned weights
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval().to(DEVICE)
print("✅ Model loaded.")

# =============== IMAGE INFERENCE ============
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE, torch.float16)
    pixel_values = inputs["pixel_values"]

    generated_ids = model.generate(pixel_values=pixel_values, max_length=35)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

# =============== TEST IT ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--web', action='store_true', help='Run web frontend')
    args = parser.parse_args()

    if args.web:
        from flask import Flask, request, jsonify, send_from_directory
        import tempfile

        app = Flask(__name__)

        @app.route("/")
        def index():
            return send_from_directory(".", "frontend.html")

        @app.route("/predict", methods=["POST"])
        def predict():
            if "image" not in request.files:
                return jsonify({"error": "No image uploaded"}), 400
            file = request.files["image"]
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                file.save(tmp.name)
                caption = generate_caption(tmp.name)
            return jsonify({"caption": caption})

        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        # Change this to your desired image
        TEST_IMAGE = "frame_30.jpg"  # <-- Replace with actual path
        if not os.path.exists(TEST_IMAGE):
            raise FileNotFoundError(f"{TEST_IMAGE} not found.")

        print(f"🖼️ Running inference on: {TEST_IMAGE}")
        caption = generate_caption(TEST_IMAGE)
        print(f"📝 Generated Caption: {caption}")
