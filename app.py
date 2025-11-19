from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont
import requests
import io
import os
import random
import string

app = Flask(__name__)

# Ensure there's a 'results' folder for saving final images
os.makedirs("results", exist_ok=True)


def download_quicksand_font():
    font_url = "https://fonts.gstatic.com/s/quicksand/v30/6xK-dSZaM9iE8KbpRA_LJ3z8mH9BOJvgkBgv58a-xw.ttf"
    font_size = 55
    try:
        response = requests.get(font_url)
        if response.status_code == 200:
            font_data = io.BytesIO(response.content)
            return ImageFont.truetype(font_data, font_size)
    except:
        pass

    try:
        return ImageFont.truetype("Quicksand-Bold.ttf", font_size)
    except:
        try:
            return ImageFont.truetype("arial.ttf", font_size)
        except:
            return ImageFont.load_default()


def resize_image(image, target_size):
    target_width, target_height = target_size
    original_width, original_height = image.size
    target_aspect = target_width / target_height
    image_aspect = original_width / original_height

    if image_aspect > target_aspect:
        new_width = target_width
        new_height = int(target_width / image_aspect)
        x_offset, y_offset = 0, (target_height - new_height) // 2
    else:
        new_height = target_height
        new_width = int(target_height * image_aspect)
        x_offset, y_offset = (target_width - new_width) // 2, 0

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized, (x_offset, y_offset)


def split_text_into_two(text):
    words = text.split()
    total_words = len(words)
    mid_index = total_words // 2
    part1 = " ".join(words[:mid_index])
    part2 = " ".join(words[mid_index:])
    return part1, part2


def create_centered_text(draw, text_parts, font, image_size, y_offset=0):
    image_width = image_size[0]
    top_text, bottom_text = text_parts

    top_bbox = draw.textbbox((0, 0), top_text, font=font)
    top_w = top_bbox[2] - top_bbox[0]
    top_h = top_bbox[3] - top_bbox[1]

    bottom_bbox = draw.textbbox((0, 0), bottom_text, font=font)
    bottom_w = bottom_bbox[2] - bottom_bbox[0]
    bottom_h = bottom_bbox[3] - bottom_bbox[1]

    spacing = 1
    total_h = top_h + bottom_h + spacing

    center_y = image_size[1] // 7 + y_offset
    top_y = center_y - total_h // 2
    bottom_y = top_y + top_h + spacing

    draw.text(((image_width - top_w) // 2, top_y),
              top_text,
              font=font,
              fill='white')
    draw.text(((image_width - bottom_w) // 2, bottom_y),
              bottom_text,
              font=font,
              fill='white')

    return total_h


def create_composite_image(
        background_path,
        overlay_image,  # now an Image object
        output_path,
        text_parts,
        output_size=(850, 750),
        text_y_offset=0):
    try:
        background = Image.open(background_path)
        if background.mode != 'RGB':
            background = background.convert('RGB')
        resized_bg, bg_off = resize_image(background, output_size)
        composite = Image.new('RGB', output_size)
        composite.paste(resized_bg, bg_off)

        if overlay_image.mode != 'RGBA':
            overlay_image = overlay_image.convert('RGBA')
        overlay_max_size = (output_size[0], output_size[0] // 3)
        resized_overlay, overlay_off = resize_image(overlay_image,
                                                    overlay_max_size)
        overlay_x = (output_size[0] - resized_overlay.size[0]) // 2
        overlay_y = (output_size[1] - resized_overlay.size[1]) // 2
        composite.paste(resized_overlay, (overlay_x, overlay_y),
                        resized_overlay)

        draw = ImageDraw.Draw(composite)
        font = download_quicksand_font()
        create_centered_text(draw, text_parts, font, output_size,
                             text_y_offset)

        composite.save(output_path, 'JPEG', quality=95)
        return True, "Composite created successfully!"
    except Exception as e:
        return False, f"Error creating composite: {str(e)}"


@app.route("/", methods=["GET"])
def home():
    return "Hello! This endpoint uses a super background + a URL overlay image."


@app.route("/process-image", methods=["POST"])
def process_image():
    try:
        # Get the overlay image from an URL string
        image_url = request.form.get("image_url")
        if not image_url:
            return jsonify({"error": "No 'image_url' provided"}), 400

        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify(
                {"error": f"Failed to download image from {image_url}"}), 400

        overlay_image = Image.open(io.BytesIO(response.content))

        # Retrieve the text and split it
        text = request.form.get("text", "Default text for overlay image")
        text_parts = split_text_into_two(text)

        background_path = "background.jpg"

        final_path = "results/final_image.jpg"

        success, msg = create_composite_image(background_path=background_path,
                                              overlay_image=overlay_image,
                                              output_path=final_path,
                                              text_parts=text_parts,
                                              output_size=(1640, 840),
                                              text_y_offset=-50)

        if not success:
            return jsonify({"error": msg}), 500

        return send_file(final_path, mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

