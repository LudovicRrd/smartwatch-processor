import os
import torch
import torch.nn.functional as F
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from flask import Flask, request, send_file, jsonify
import replicate
import tempfile
import uuid

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

app = Flask(__name__)

# Use environment variable for port (Render requirement)
PORT = int(os.environ.get('PORT', 8000))

# -----------------------------------------------------------------
# 1) DETECTION AND MASK GENERATION USING CLIPSeg
# -----------------------------------------------------------------
def create_screen_mask(
        image_path: str,
        prompt: str = "Detect only the active display area (screen). Focus solely on the digital display area, which may be circular, square, or rectangular",
        threshold: float = 0.8,
        alpha: float = 0.1):
    """
    Generate a mask using CLIPSeg for the smartwatch screen, apply thresholding,
    and return the path to the saved mask image.
    """
    image = Image.open(image_path).convert("RGB")

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    inputs_img = processor.image_processor(images=image,
                                           do_resize=True,
                                           size=(352, 352),
                                           return_tensors="pt")
    inputs_txt = processor.tokenizer([prompt], return_tensors="pt", truncation=True)
    inputs = {
        "pixel_values": inputs_img["pixel_values"],
        "input_ids": inputs_txt["input_ids"],
        "attention_mask": inputs_txt["attention_mask"]
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    orig_w, orig_h = image.size
    upsampled = F.interpolate(logits.unsqueeze(1),
                              size=(orig_h, orig_w),
                              mode="bicubic",
                              align_corners=False)

    mask_np = upsampled[0, 0].cpu().numpy()
    binary = (mask_np > threshold).astype(np.uint8) * 255
    inverted = 255 - binary

    mask_img = Image.fromarray(inverted, mode="L")
    # Use temporary file with unique name
    mask_path = f"/tmp/mask_{uuid.uuid4().hex}.png"
    mask_img.save(mask_path)
    return mask_path

# -----------------------------------------------------------------
# 2) REPLICATE LOGIC (to replace watch screen)
# -----------------------------------------------------------------
def replace_watch_screen(image_path, mask_path, prompt, output_path):
    """
    Replace the smartwatch screen using ideogram-v2 from Replicate.
    """
    try:
        # Ensure API token is set
        if not os.getenv("REPLICATE_API_TOKEN"):
            raise ValueError("REPLICATE_API_TOKEN environment variable not set")
            
        output = replicate.run(
            "ideogram-ai/ideogram-v2",
            input={
                "prompt": prompt,
                "image": open(image_path, "rb"),
                "mask": open(mask_path, "rb"),
                "guidance_scale": 7.5,
                "negative_prompt": "blurry, distorted, low quality, unrealistic, wrong perspective",
                "num_inference_steps": 50,
                "scheduler": "K_EULER"
            }
        )

        # Extract URL and download
        url = None
        if hasattr(output, "url"):
            url = output.url
        elif isinstance(output, str) and output.startswith("http"):
            url = output

        if url:
            resp = requests.get(url, stream=True)
            if resp.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(1024):
                        f.write(chunk)
                return True
        return False

    except Exception as e:
        print(f"Error during Replicate run: {e}")
        return False

# -----------------------------------------------------------------
# 3) FLASK ROUTES
# -----------------------------------------------------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "Service is up and running", "health": "OK"})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/process-image', methods=['POST'])
def process_image():
    """
    1. Receives an image (smartwatch) via POST (multipart/form-data or JSON URL).
    2. Looks for 'mask_url' in form data or JSON body.
    3. If a mask_url is provided, tries to download it; on failure, falls back to CLIPSeg.
    4. If no mask_url, generates mask via CLIPSeg.
    5. Replaces watch screen via Replicate and returns the result.
    """
    # Create temporary files with unique names
    image_path = f"/tmp/image_{uuid.uuid4().hex}.jpg"
    output_path = f"/tmp/result_{uuid.uuid4().hex}.jpg"
    mask_path = None
    
    try:
        # 1) Save uploaded image
        if 'file' in request.files:
            image_file = request.files['file']
            image_file.save(image_path)
        else:
            return jsonify({"error": "No image file provided."}), 400

        # 2) Fetch mask_url from form or JSON
        mask_url = request.form.get('mask_url') or (request.get_json(silent=True) or {}).get('mask_url')

        # 3) Download or generate mask
        if mask_url:
            mask_path = f"/tmp/mask_{uuid.uuid4().hex}.png"
            try:
                resp = requests.get(mask_url, stream=True)
                if resp.status_code == 200:
                    with open(mask_path, 'wb') as f:
                        for chunk in resp.iter_content(1024):
                            f.write(chunk)
                else:
                    # fallback to CLIPSeg if download fails
                    prompt = request.form.get('prompt', '')
                    mask_path = create_screen_mask(image_path, prompt)
            except Exception:
                prompt = request.form.get('prompt', '')
                mask_path = create_screen_mask(image_path, prompt)
        else:
            # No URL: generate mask
            prompt = request.form.get('prompt', '')
            mask_path = create_screen_mask(image_path, prompt)

        # 4) Prepare prompt and output path
        replacement_prompt = request.form.get('prompt', 'change the smartwatch screen for a full black smartwatch')
        
        # 5) Replace screen
        success = replace_watch_screen(image_path, mask_path, replacement_prompt, output_path)

        # 6) Return
        result_file = output_path if success else image_path
        
        # Clean up temporary files after sending response
        def cleanup():
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                if mask_path and os.path.exists(mask_path):
                    os.remove(mask_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
            except:
                pass
        
        response = send_file(result_file, mimetype='image/jpeg')
        response.call_on_close(cleanup)
        return response

    except Exception as e:
        # Clean up on error
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
            if mask_path and os.path.exists(mask_path):
                os.remove(mask_path)
            if os.path.exists(output_path):
                os.remove(output_path)
        except:
            pass
        
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------------
# 4) MAIN
# -----------------------------------------------------------------
if __name__ == '__main__':
    # Use PORT from environment (required for Render)
    app.run(host='0.0.0.0', port=PORT, debug=False)