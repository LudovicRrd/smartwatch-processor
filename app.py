from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont
import requests
import io
import os

app = Flask(__name__)

# Ensure there's a 'results' folder for saving final images
os.makedirs("results", exist_ok=True)


def parse_rgb_color(value):
    """Parse #RGB, #RRGGBB, or 'r,g,b' into an (R, G, B) tuple."""
    if value is None:
        raise ValueError("missing color")
    s = str(value).strip()
    if not s:
        raise ValueError("empty color")
    if s.startswith("#"):
        hexv = s[1:]
        if len(hexv) == 3:
            r, g, b = (int(c, 16) * 17 for c in hexv)
        elif len(hexv) == 6:
            r = int(hexv[0:2], 16)
            g = int(hexv[2:4], 16)
            b = int(hexv[4:6], 16)
        else:
            raise ValueError("invalid hex color")
        return (r, g, b)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("color must be #hex or r,g,b")
    return tuple(int(float(x)) for x in parts)


def build_background_from_colors(
        output_size,
        color_top,
        color_bottom,
        bottom_band_ratio=1 / 3,
        logo_image=None,
        logo_max_height_fraction=0.55):
    """
    Two horizontal bands (replaces a static background image):
    - color_top (bg_color_1): upper ~2/3 of the canvas
    - color_bottom (bg_color_2): lower ~1/3 (logo sits bottom-right in this band).
    """
    w, h = output_size
    split_y = max(1, int(h * (1 - bottom_band_ratio)))
    img = Image.new("RGB", output_size, color_top)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, split_y, w, h], fill=color_bottom)

    if logo_image is None:
        return img

    logo_image = logo_image.copy()
    if logo_image.mode != "RGBA":
        logo_image = logo_image.convert("RGBA")

    lw, lh = logo_image.size
    band_h = h - split_y
    max_h = max(1, int(band_h * logo_max_height_fraction))
    max_w = max(1, int(w * 0.45))
    scale = min(max_h / lh, max_w / lw, 1.0)
    nw, nh = max(1, int(lw * scale)), max(1, int(lh * scale))
    logo_r = logo_image.resize((nw, nh), Image.Resampling.LANCZOS)

    pad = max(20, int(w * 0.02))
    x = w - nw - pad
    y = h - nh - pad
    img.paste(logo_r, (x, y), logo_r)
    return img


def load_background_image(background_path_or_image, output_size):
    """Open path or use PIL Image, convert to RGB, letterbox to output_size."""
    if isinstance(background_path_or_image, str):
        background = Image.open(background_path_or_image)
    else:
        background = background_path_or_image
    if background.mode != "RGB":
        background = background.convert("RGB")
    resized_bg, bg_off = resize_image(background, output_size)
    composite = Image.new("RGB", output_size)
    composite.paste(resized_bg, bg_off)
    return composite


def download_quicksand_font(font_size=70):
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


def create_left_text_with_subtitle_and_cta(
        base_image,  # RGB image
        text_parts,
        subtitle,
        cta_text,
        font,
        image_size,
        y_offset=0,
        title_bg_opacity=0.8,
        subtitle_bg_opacity=0.8,
        cta_bg_opacity=0.8,
        corner_radius=15,
        cta_corner_radius=60):
    """
    Draw two title lines (orange overlay) left-aligned,
    then subtitle (green overlay) and CTA button (red overlay) below. [web:3][web:21][web:51]
    """
    draw = ImageDraw.Draw(base_image, "RGBA")

    image_width = image_size[0]
    top_text, bottom_text = text_parts

    # Left margin for all text
    margin_x = 260

    # Measure title lines
    top_bbox = draw.textbbox((0, 0), top_text, font=font)
    top_w = top_bbox[2] - top_bbox[0]
    top_h = top_bbox[3] - top_bbox[1]

    bottom_bbox = draw.textbbox((0, 0), bottom_text, font=font)
    bottom_w = bottom_bbox[2] - bottom_bbox[0]
    bottom_h = bottom_bbox[3] - bottom_bbox[1]

    # Gap between the two title lines
    spacing_title_lines = 40
    total_h = top_h + bottom_h + spacing_title_lines

    # Vertical position of title block
    center_y = image_size[1] // 7 + y_offset
    top_y = center_y - total_h // 2
    bottom_y = top_y + top_h + spacing_title_lines

    # Colors (RGBA)
    title_bg_color = (56, 56, 56, int(255 * title_bg_opacity)
                      )  # orange for title
    subtitle_bg_color = (56, 56, 56, int(255 * subtitle_bg_opacity)
                         )  # green for subtitle
    cta_bg_color = (68, 159, 119, int(255 * cta_bg_opacity))  # red for CTA

    pad_x = 50
    pad_y = 50

    # First title line (orange)
    if top_text.strip():
        rect_x0 = max(0, margin_x - pad_x)
        rect_y0 = max(0, top_y - pad_y // 5)
        rect_x1 = min(image_width, margin_x + top_w + pad_x)
        rect_y1 = rect_y0 + top_h + pad_y

        draw.rounded_rectangle([(rect_x0, rect_y0), (rect_x1, rect_y1)],
                               radius=corner_radius,
                               fill=title_bg_color)

    draw.text((margin_x, top_y), top_text, font=font, fill='white')

    # Second title line (also orange)
    if bottom_text.strip():
        rect_x0 = max(0, margin_x - pad_x)
        rect_y0 = max(0, bottom_y - pad_y // 5)
        rect_x1 = min(image_width, margin_x + bottom_w + pad_x)
        rect_y1 = rect_y0 + bottom_h + pad_y

        draw.rounded_rectangle([(rect_x0, rect_y0), (rect_x1, rect_y1)],
                               radius=corner_radius,
                               fill=title_bg_color)

    draw.text((margin_x, bottom_y), bottom_text, font=font, fill='white')

    # Subtitle below title block, smaller font, green overlay
    subtitle_y = bottom_y + bottom_h
    sub_h = 0

    if subtitle.strip():
        sub_font_size = max(50, font.size // 2.2)
        sub_font = download_quicksand_font(sub_font_size)

        sub_bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sub_w = sub_bbox[2] - sub_bbox[0]
        sub_h = sub_bbox[3] - sub_bbox[1]

        # More space between title and subtitle
        subtitle_spacing = 100
        subtitle_y = bottom_y + bottom_h + subtitle_spacing

        sub_pad_x = 40
        sub_pad_y = 30

        rect_x0 = max(0, margin_x - sub_pad_x)
        rect_y0 = max(0, subtitle_y - sub_pad_y // 3)
        rect_x1 = min(image_width, margin_x + sub_w + sub_pad_x)
        rect_y1 = rect_y0 + sub_h + sub_pad_y

        draw.rounded_rectangle([(rect_x0, rect_y0), (rect_x1, rect_y1)],
                               radius=corner_radius,
                               fill=subtitle_bg_color)

        draw.text((margin_x, subtitle_y),
                  subtitle,
                  font=sub_font,
                  fill='white')

    # CTA button under subtitle, red pill, slightly larger font than subtitle
    if cta_text.strip():
        if subtitle.strip():
            base_font_size = max(50, font.size // 2.2)
        else:
            base_font_size = max(50, font.size // 2.2)

        cta_font_size = max(50, font.size // 2.2)  # a bit bigger than subtitle
        cta_font = download_quicksand_font(cta_font_size)

        cta_bbox = draw.textbbox((0, 0), cta_text, font=cta_font)
        cta_w = cta_bbox[2] - cta_bbox[0]
        cta_h = cta_bbox[3] - cta_bbox[1]

        cta_spacing = 60  # space between subtitle and CTA
        if subtitle.strip():
            cta_y = subtitle_y + sub_h + cta_spacing
        else:
            cta_y = bottom_y + bottom_h + cta_spacing

        cta_pad_x = 40
        cta_pad_y = 40

        rect_x0 = max(0, margin_x - cta_pad_x)
        rect_y0 = max(0, cta_y - cta_pad_y // 4)
        rect_x1 = min(image_width, margin_x + cta_w + cta_pad_x)
        rect_y1 = rect_y0 + cta_h + cta_pad_y

        draw.rounded_rectangle(
            [(rect_x0, rect_y0), (rect_x1, rect_y1)],
            radius=cta_corner_radius,  # very rounded = pill‑like
            fill=cta_bg_color)

        draw.text((margin_x, cta_y), cta_text, font=cta_font, fill='white')

    return total_h


def create_composite_image(
        background_path_or_image,
        overlay_image,  # Image object
        output_path,
        text_parts,
        subtitle,
        cta_text,
        output_size=(1640, 840),
        text_y_offset=0,
        overlay_opacity=0.1):
    """
    Compose background (PIL Image or path), uploaded image centered (contain),
    same layout as Hostinger main.py: max box width × (width/3), then dim + text.
    """
    try:
        composite = load_background_image(background_path_or_image, output_size)

        if overlay_image.mode != 'RGBA':
            overlay_image = overlay_image.convert('RGBA')

        # Hostinger-style: full photo visible, max height = width/3, centered on canvas
        overlay_max_size = (output_size[0], output_size[0] // 3)
        resized_overlay, _ = resize_image(overlay_image, overlay_max_size)
        overlay_x = (output_size[0] - resized_overlay.size[0]) // 2
        overlay_y = (output_size[1] - resized_overlay.size[1]) // 2
        composite.paste(resized_overlay, (overlay_x, overlay_y),
                        resized_overlay)

        if overlay_opacity > 0:
            ow, oh = resized_overlay.size
            overlay_layer = Image.new(
                'RGBA', (ow, oh),
                color=(0, 0, 0, int(255 * overlay_opacity)))
            composite.paste(overlay_layer, (overlay_x, overlay_y), overlay_layer)

        # Title (2 lines, left) + subtitle + CTA
        font = download_quicksand_font(80)
        create_left_text_with_subtitle_and_cta(base_image=composite,
                                               text_parts=text_parts,
                                               subtitle=subtitle,
                                               cta_text=cta_text,
                                               font=font,
                                               image_size=output_size,
                                               y_offset=text_y_offset,
                                               title_bg_opacity=0.7,
                                               subtitle_bg_opacity=0.7,
                                               cta_bg_opacity=0.7,
                                               corner_radius=15,
                                               cta_corner_radius=60)

        composite.save(output_path, 'JPEG', quality=95)
        return True, "Composite created successfully!"
    except Exception as e:
        return False, f"Error creating composite: {str(e)}"


@app.route("/", methods=["GET"])
def home():
    return (
        "POST /process-image: image_url (required); bg_color_1 (top ~2/3), bg_color_2 (bottom ~1/3), "
        "#hex or r,g,b; optional logo_url; text, subtitle, cta, overlay_opacity. Output 1640×840 JPEG."
    )


@app.route("/process-image", methods=["POST"])
def process_image():
    try:
        image_url = request.form.get("image_url")
        if not image_url:
            return jsonify({"error": "No 'image_url' provided"}), 400

        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify(
                {"error": f"Failed to download image from {image_url}"}), 400

        overlay_image = Image.open(io.BytesIO(response.content))

        # Main title text (split into 2 lines)
        text = request.form.get("text", "Default text for overlay image")
        text_parts = split_text_into_two(text)

        # Subtitle from request
        subtitle = request.form.get("subtitle", "")

        # CTA text from request
        cta_text = request.form.get("cta", "")

        overlay_opacity = float(request.form.get("overlay_opacity", 0.1))

        c1 = request.form.get("bg_color_1")
        c2 = request.form.get("bg_color_2")
        logo_url = (request.form.get("logo_url") or "").strip()

        if not (c1 and str(c1).strip() and c2 and str(c2).strip()):
            return jsonify(
                {"error": "bg_color_1 and bg_color_2 are required (#hex or r,g,b)"}
            ), 400

        try:
            color_top = parse_rgb_color(c1)
            color_bottom = parse_rgb_color(c2)
        except ValueError as e:
            return jsonify({"error": f"Invalid background color: {e}"}), 400

        logo_img = None
        if logo_url:
            try:
                lr = requests.get(logo_url, timeout=30)
                if lr.status_code != 200:
                    return jsonify(
                        {"error": f"Failed to download logo from {logo_url}"}
                    ), 400
                logo_img = Image.open(io.BytesIO(lr.content))
            except Exception as e:
                return jsonify({"error": f"Logo error: {str(e)}"}), 400

        out_sz = (1640, 840)
        background_source = build_background_from_colors(
            out_sz,
            color_top,
            color_bottom,
            logo_image=logo_img,
        )

        final_path = "results/final_image.jpg"

        success, msg = create_composite_image(
            background_path_or_image=background_source,
            overlay_image=overlay_image,
            output_path=final_path,
            text_parts=text_parts,
            subtitle=subtitle,
            cta_text=cta_text,
            output_size=out_sz,
            text_y_offset=-50,
            overlay_opacity=overlay_opacity,
        )

        if not success:
            return jsonify({"error": msg}), 500

        return send_file(final_path, mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
