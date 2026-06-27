import io
import random

import numpy as np
import pandas as pd
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    session,
    send_file
)

from PIL import Image

from generator import QGANGenerator

# ============================================================
# Flask
# ============================================================

app = Flask(__name__)

# Change this to something random before deployment
app.secret_key = "replace_this_with_a_random_secret_key"

# ============================================================
# Load Generator
# ============================================================

generator = QGANGenerator("generator.pth")

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("optdigits.tra")


# ============================================================
# Helper Functions
# ============================================================

def random_real_image():

    row = df.sample(1).iloc[0]

    image = row.iloc[:-1].to_numpy(dtype=np.float32)

    image = image.reshape(8, 8)

    image = image / 16.0

    label = int(row.iloc[-1])

    return image, label


def fake_image_for_label(label):

    image = generator.generate(label)

    return image


def numpy_to_png(image):

    image = np.clip(image, 0, 1)

    image = (image * 255).astype(np.uint8)

    pil = Image.fromarray(image)

    buffer = io.BytesIO()

    pil.save(buffer, format="PNG")

    buffer.seek(0)

    return buffer


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():

    if "streak" not in session:
        session["streak"] = 0

    if "best" not in session:
        session["best"] = 0

    if "games" not in session:
        session["games"] = 0

    if "correct" not in session:
        session["correct"] = 0

    return render_template("index.html")


# ============================================================
# New Round
# ============================================================

@app.route("/new_round")
def new_round():

    real_image, real_label = random_real_image()

    fake_image = fake_image_for_label(real_label)

    fake_label = real_label

    fake_left = random.choice([True, False])

    session["fake_left"] = fake_left

    session["fake_image"] = fake_image.tolist()

    session["real_image"] = real_image.tolist()

    session["fake_label"] = fake_label

    session["real_label"] = real_label

    return jsonify({

        "left": "/image/left",

        "right": "/image/right",

        "streak": session["streak"],

        "best": session["best"],

        "accuracy": (
            0
            if session["games"] == 0
            else round(
                100 * session["correct"] / session["games"],
                1
            )
        )

    })


# ============================================================
# Serve Images
# ============================================================

@app.route("/image/<side>")
def image(side):

    fake_left = session["fake_left"]

    if side == "left":

        if fake_left:
            img = np.array(session["fake_image"])

        else:
            img = np.array(session["real_image"])

    else:

        if fake_left:
            img = np.array(session["real_image"])

        else:
            img = np.array(session["fake_image"])

    return send_file(
        numpy_to_png(img),
        mimetype="image/png"
    )


# ============================================================
# Guess
# ============================================================

@app.route("/guess", methods=["POST"])
def guess():

    data = request.json

    guess = data["choice"]

    fake_left = session["fake_left"]

    if fake_left:
        correct_choice = "left"
    else:
        correct_choice = "right"

    correct = (guess == correct_choice)

    session["games"] += 1

    if correct:

        session["correct"] += 1

        session["streak"] += 1

        if session["streak"] > session["best"]:
            session["best"] = session["streak"]

    else:

        session["streak"] = 0

    return jsonify({

        "correct": correct,

        "fake_side": correct_choice,

        "real_side": (
            "right"
            if correct_choice == "left"
            else "left"
        ),

        "fake_label": session["fake_label"],

        "real_label": session["real_label"],

        "streak": session["streak"],

        "best": session["best"],

        "accuracy": round(
            100 * session["correct"] / session["games"],
            1
        )

    })


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )