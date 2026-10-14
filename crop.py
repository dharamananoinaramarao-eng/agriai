import os
import base64
from io import BytesIO
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    Groq,
    NotFoundError,
    RateLimitError,
)
from PIL import Image, UnidentifiedImageError

load_dotenv(Path(__file__).with_name("crop.env"))

app = Flask(__name__, template_folder=".")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
client = Groq(api_key=api_key) if api_key else None

@app.route("/")
def home():
    return render_template("crop.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if not client:
        return jsonify({
            "error": "GROQ_API_KEY is missing from your crop.env file."
        }), 500

    file = request.files.get("photo")

    if not file:
        return jsonify({"error": "Upload or capture a photo first."}), 400

    try:
        raw = file.read()

        if not raw:
            return jsonify({"error": "The image is empty."}), 400

        image = Image.open(BytesIO(raw))
        image.verify()

        image = Image.open(BytesIO(raw))
        image.load()

        if image.width < 50 or image.height < 50:
            return jsonify({
                "error": "Please upload a clearer, larger photo."
            }), 400

        image.thumbnail((1024, 1024))
        buffer = BytesIO()
        image.convert("RGB").save(
            buffer, format="JPEG", quality=85
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "Invalid image file."}), 400

    prompt = """
    You are an agricultural crop-photo assistant.

    Examine the provided image and explain:
    1. Likely crop or plant, if identifiable.
    2. Visible symptoms.
    3. Possible disease, pest, or nutrient issue.
    4. How certain the assessment is and what else could cause it.
    5. Practical next steps for the farmer.
    6. Treatment guidance, only when sufficiently supported.

    If the photo is unclear or the diagnosis is uncertain, say so.
    Do not invent a diagnosis or provide unsupported pesticide doses.
    Recommend confirmation by an agricultural expert before applying
    pesticides. AI image analysis is preliminary, not a laboratory test.
    """

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/jpeg;base64," + encoded
                                )
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_completion_tokens=800
        )

        details = response.choices[0].message.content

        return jsonify({
            "details": details or "No analysis was returned."
        })

    except AuthenticationError:
        app.logger.exception("Groq authentication failed")
        return jsonify({
            "error": "Groq rejected the API key. Check that the key is valid and active."
        }), 502

    except RateLimitError:
        app.logger.exception("Groq rate limit reached")
        return jsonify({
            "error": "Groq rate limit reached. Please wait and try again."
        }), 502

    except NotFoundError:
        app.logger.exception("Groq model not found")
        return jsonify({
            "error": "The configured Groq model is unavailable. Check the model name and access."
        }), 502

    except APIConnectionError:
        app.logger.exception("Could not connect to Groq")
        return jsonify({
            "error": "Could not connect to Groq. Check your internet connection and try again."
        }), 502

    except BadRequestError:
        app.logger.exception("Groq rejected the analysis request")
        return jsonify({
            "error": "Groq rejected the request. Check that the image and model are supported."
        }), 502

    except Exception:
        app.logger.exception("AI image analysis error")
        return jsonify({
            "error": (
                "AI request failed. Check the API key, model availability, "
                "internet connection, and API usage limits."
            )
        }), 502

if __name__ == "__main__":
    app.run(debug=True)