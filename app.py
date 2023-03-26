import argparse
import io
import os
from PIL import Image

import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])

        results.render()  # updates results.imgs with boxes and labels

        img_savename = f"static/image/image0.jpg"
        Image.fromarray(results.ims[0]).save(img_savename)

        return redirect('/result')

    return render_template("index.html")

@app.route("/result")
def result():
    full_filename = os.path.join('static', 'image/image0.jpg')
    return render_template("result.html", result_image = full_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drowned Victim Finder")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('yolov5', 'custom', path='last.pt', source='local')
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
