from flask import Flask, request, jsonify, send_file, make_response
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import os
import requests
import tempfile
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# Color mapping with transparency
CLASS_COLORS = {
    "Glioma": (0, 0, 255, 100),        # Blue
    "Meningioma": (255, 165, 0, 100),  # Orange
    "No_Tumor": (0, 255, 0, 100),      # Green
    "Pituitary": (255, 0, 0, 100)      # Red
}

# Roboflow API client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="LeLMFj429X9iPWbNH7GK"
)

# Draw predictions on image
def overlay_predictions(image, predictions):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for pred in predictions:
        points = [(p["x"], p["y"]) for p in pred["points"]]
        label = pred["class"]
        confidence = int(pred["confidence"] * 100)
        text = f"{label} {confidence}%"
        color = CLASS_COLORS.get(label, (255, 255, 255, 100))

        draw.polygon(points, fill=color)

        top_point = min(points, key=lambda p: p[1])

        text_size = draw.textbbox((0, 0), text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        text_position = (top_point[0], top_point[1] - text_height - 4)

        rect_coords = [
            text_position,
            (text_position[0] + text_width + 6, text_position[1] + text_height + 4)
        ]
        draw.rectangle(rect_coords, fill=color[:3] + (200,))
        draw.text((text_position[0] + 3, text_position[1] + 2), text, fill=(0, 0, 0), font=font)

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

@app.route("/process", methods=["POST"])
def process_image():
    data = request.get_json()
    if not data or "image_url" not in data:
        return jsonify({"error": "Missing 'image_url' in JSON body"}), 400

    image_url = data["image_url"]

    try:
        response = requests.get(image_url)
        response.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400

    try:
        image = Image.open(BytesIO(response.content)).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            image.save(temp.name)
            temp_path = temp.name

        result_json = client.run_workflow(
            workspace_name="braintumordetectionandsegementation",
            workflow_id="custom-workflow-3",
            images={"image": temp_path},
            use_cache=True
        )

        os.remove(temp_path)

        predictions = result_json[0]["predictions"]["predictions"]
        annotated = overlay_predictions(image, predictions)

        result_text = "\n".join(
            [f"{pred['class']} {int(pred['confidence'] * 100)}%" for pred in predictions]
        )

        output_buffer = BytesIO()
        annotated.save(output_buffer, format="JPEG")
        output_buffer.seek(0)

        response = make_response(send_file(
            output_buffer,
            mimetype='image/jpeg',
            as_attachment=False,
            download_name='annotated.jpg'
        ))
        response.headers["X-Result"] = result_text
        response.headers["X-Message"] = "response done successfully"
        return response

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Brain tumor segmentation API is running!"})
