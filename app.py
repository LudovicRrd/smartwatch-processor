from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont
import requests
import io
import os

app = Flask(__name__)

# Ensure there's a 'results' folder for saving final images
os.makedirs("results", exist_ok=True)


def download_quicksand_font(font_size=80):
    font_url = "https://fonts.gstatic.com/s/quicksand/v30/6xK-dSZaM9iE8KbpRA_LJ3z8mH9BOJvgkBgv58a-xw.ttf"
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


def resize_to_full_width(image, target_width):
    original_width, original_height = image.size
    if original_width == 0:
        return image, (0, 0)
    ratio = target_width / original_width
    new_height = int(original_height * ratio + 0.5)
    resized = image.resize((target_width, new_height),
                           Image.Resampling.LANCZOS)
    return resized, (0, 0)


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


def split_text_into_lines(text, font, max_width, num_lines=5):
    if not text or not text.strip():
        return [""] * num_lines
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return [""] * num_lines
    part_size = max(1, total_words // num_lines)
    parts = []
    for i in range(num_lines):
        start = i * part_size
        end = start + part_size if i < num_lines - 1 else total_words
        line_words = words[start:end]
        line_text = " ".join(line_words)
        # Truncate to fit width
        while font.getlength(line_text) > max_width and len(line_words) > 0:
            if len(line_words) > 1:
                line_words.pop()
                line_text = " ".join(line_words)
            else:
                line_text = line_words[
                    0][:int(max_width / (font.getlength("A") or 1) *
                            0.8)] + "..." if line_words else "..."
                break
        parts.append(line_text if line_text else " ")
        if i >= num_lines - 1:
            break
    return parts


def create_top_left_text_over_image(
        base_image,  # RGB image
        text,
        subtitle,
        font_size_main=90,
        image_size=(1080, 1920),
        margin=50,
        line_spacing=0,
        title_bg_opacity=0.6,
        subtitle_bg_opacity=0.6,
        corner_radius=25  # new: rounded corner radius in px
):
    """
    Draw title + subtitle, with adaptive semi-transparent strips behind each line. [web:1][web:8]
    """
    max_w = image_size[0] - 2 * margin

    # Use RGBA drawing mode on an RGB image so alpha blends correctly. [web:8]
    draw = ImageDraw.Draw(base_image, "RGBA")

    # Main title font and lines (5 lines max)
    font_main = download_quicksand_font(font_size_main)
    main_lines = split_text_into_lines(text, font_main, max_w, num_lines=5)

    # Compute line height for main
    ascent_main, descent_main = font_main.getmetrics()
    line_height_main = ascent_main + descent_main + line_spacing

    # Colors (RGBA) for background strips
    title_bg_color = (255, 140, 0, int(255 * title_bg_opacity)
                      )  # orange [web:1]
    subtitle_bg_color = (68, 159, 119, int(255 * subtitle_bg_opacity)
                         )  # green [web:1]

    # Padding around text in the strip
    pad_x = 30
    pad_y = 20

    y = margin

    # Draw title lines with orange background
    for line in main_lines:
        if line.strip():
            text_width = font_main.getlength(line)
            rect_x0 = max(0, margin - pad_x)
            rect_y0 = max(0, y - pad_y // 2)
            rect_x1 = min(image_size[0], margin + int(text_width) + pad_x)
            rect_y1 = min(image_size[1], y + line_height_main + pad_y // 2)

            draw.rounded_rectangle([(rect_x0, rect_y0), (rect_x1, rect_y1)],
                                   radius=corner_radius,
                                   fill=title_bg_color)

        draw.text((margin, y), line, font=font_main, fill='white')
        y += line_height_main

    # Subtitle below main (up to 2 lines, half size)
    if subtitle.strip():
        font_sub = download_quicksand_font(font_size_main // 2)
        sub_lines = split_text_into_lines(subtitle,
                                          font_sub,
                                          max_w,
                                          num_lines=2)

        # Extra spacing between title and subtitle (10px for separation)
        y += 10

        # Compute line height for subtitle
        ascent_sub, descent_sub = font_sub.getmetrics()
        line_height_sub = ascent_sub + descent_sub + line_spacing

        # Draw subtitle lines with green background
        for line in sub_lines:
            if line.strip():
                text_width = font_sub.getlength(line)
                rect_x0 = max(0, margin - pad_x)
                rect_y0 = max(0, y - pad_y // 2)
                rect_x1 = min(image_size[0], margin + int(text_width) + pad_x)
                rect_y1 = min(image_size[1], y + line_height_sub + pad_y // 2)

                draw.rounded_rectangle([(rect_x0, rect_y0),
                                        (rect_x1, rect_y1)],
                                       radius=corner_radius,
                                       fill=subtitle_bg_color)

            draw.text((margin, y), line, font=font_sub, fill='white')
            y += line_height_sub

    return y  # Total height used by text


def create_composite_image(
        background_path,
        overlay_image,  # Image object
        output_path,
        text,
        subtitle,
        output_size=(1080, 1920),
        overlay_opacity=0.4  # 0.0 to 1.0 for black overlay on the photo
):
    try:
        # Load and resize background image
        background = Image.open(background_path)
        if background.mode != 'RGB':
            background = background.convert('RGB')

        resized_bg, bg_off = resize_image(background, output_size)

        # IMPORTANT: keep composite in RGB so RGBA drawing blends alpha. [web:8]
        composite = Image.new('RGB', output_size, color=(247, 247, 247))
        composite.paste(resized_bg, bg_off)

        # Prepare overlay (input image) to full width, preserving aspect
        if overlay_image.mode != 'RGBA':
            overlay_image = overlay_image.convert('RGBA')
        resized_overlay, _ = resize_to_full_width(overlay_image,
                                                  output_size[0])

        # Paste main photo at top-left (full width)
        composite.paste(resized_overlay, (0, 0), resized_overlay)

        # Add transparent black overlay over the image area (behind text). [web:1]
        overlay_height = resized_overlay.height
        overlay_layer = Image.new('RGBA', (output_size[0], overlay_height),
                                  color=(0, 0, 0, int(255 * overlay_opacity)))
        composite.paste(overlay_layer, (0, 0), overlay_layer)

        # Draw title + subtitle with their own colored strips. [web:1][web:8]
        create_top_left_text_over_image(base_image=composite,
                                        text=text,
                                        subtitle=subtitle,
                                        font_size_main=90,
                                        image_size=output_size)

        composite.save(output_path, 'JPEG', quality=95)
        return True, "Composite created successfully!"
    except Exception as e:
        return False, f"Error creating composite: {str(e)}"


@app.route("/", methods=["GET"])
def home():
    return "Hello! This endpoint creates 9:16 ads with: background, full-width image overlay, transparent black overlay on the photo, and adaptive semi-transparent orange/green strips behind title and subtitle."


@app.route("/process-image", methods=["POST"])
def process_image():
    try:
        # Get the overlay image from URL
        image_url = request.form.get("image_url")
        if not image_url:
            return jsonify({"error": "No 'image_url' provided"}), 400

        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify(
                {"error": f"Failed to download image from {image_url}"}), 400

        overlay_image = Image.open(io.BytesIO(response.content))

        # Retrieve the text and subtitle
        text = request.form.get("text", "Default title text overlay on image")
        subtitle = request.form.get("subtitle", "")

        # Optional overlay opacity parameter (0.0 to 1.0) for the black photo overlay
        overlay_opacity = float(request.form.get("overlay_opacity", 0.4))

        background_path = "background-p3.png"
        final_path = "results/final_image.jpg"

        success, msg = create_composite_image(background_path=background_path,
                                              overlay_image=overlay_image,
                                              output_path=final_path,
                                              text=text,
                                              subtitle=subtitle,
                                              output_size=(1080, 1920),
                                              overlay_opacity=overlay_opacity)

        if not success:
            return jsonify({"error": msg}), 500

        return send_file(final_path, mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
