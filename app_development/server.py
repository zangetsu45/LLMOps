from flask import Flask, request, redirect, url_for
from flask import render_template_string
import os
import threading
from werkzeug.utils import secure_filename
import gradio as gr
import subprocess
import socket
import re, time

app = Flask(__name__)
UPLOAD_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Home page: links to list models and model upload form
@app.route("/")
def index():
    return """
    <html>
    <head><title>Model Server</title></head>
    <body>
      <h1>Welcome to the Model Server</h1>
      <p><a href="/models">List Available Models</a></p>
      <h3>Upload a New Model</h3>
      <form action="/upload" method="post" enctype="multipart/form-data">
         <label>Model Definition (app.py):</label>
         <input type="file" name="model_definition"><br><br>
         <label>Model Weights (model.pth):</label>
         <input type="file" name="model_weights"><br><br>
         <label>Requirements (requirements.txt):</label>
         <input type="file" name="requirements"><br><br>
         <input type="submit" value="Upload">
      </form>
    </body>
    </html>
    """

# Endpoint to handle model uploads
@app.route("/upload", methods=["POST"])
def upload_model():
    required_files = ["model_definition", "model_weights", "requirements"]
    for file_key in required_files:
        if file_key not in request.files:
            return f"No file provided for {file_key}", 400

    model_def_file = request.files["model_definition"]
    weights_file = request.files["model_weights"]
    req_file = request.files["requirements"]

    if model_def_file.filename == "" or weights_file.filename == "" or req_file.filename == "":
        return "One or more files were not selected", 400

    # Use the uploaded model definition filename (without extension) as the model identifier.
    filename_base = os.path.splitext(secure_filename(model_def_file.filename))[0]
    model_folder = os.path.join(UPLOAD_FOLDER, filename_base)
    os.makedirs(model_folder, exist_ok=True)

    model_def_path = os.path.join(model_folder, secure_filename(model_def_file.filename))
    weights_path = os.path.join(model_folder, secure_filename(weights_file.filename))
    req_path = os.path.join(model_folder, secure_filename(req_file.filename))

    model_def_file.save(model_def_path)
    weights_file.save(weights_path)
    req_file.save(req_path)

    return redirect(url_for("list_models"))

# Endpoint to list available models
@app.route("/models", methods=["GET"])
def list_models():
    models = os.listdir(UPLOAD_FOLDER)
    model_links = "".join(
        [f'<li><a href="/model/{model}">{model}</a></li>' for model in models]
    )
    return f"""
    <html>
    <head><title>Available Models</title></head>
    <body>
      <h1>Available Models</h1>
      <ul>
        {model_links}
      </ul>
      <p><a href="/">Go Back Home</a></p>
    </body>
    </html>
    """

# Dictionary to store launched gradio interfaces and their ports.
gradio_servers = {}

@app.route("/model/<model_name>", methods=["GET"])
def model_specific(model_name):
    # Check if the model exists in our models directory.
    if model_name not in os.listdir(UPLOAD_FOLDER):
        return "Model not found", 404

    # If this model's server hasn't been started yet, create and launch it.
    if model_name not in gradio_servers:
        model_folder = os.path.join(UPLOAD_FOLDER, model_name)
        # Launch the model process without forcing a port.
        # Since we cannot modify test.py, we capture its stdout to parse the port at runtime.
        model_file = os.path.join(model_folder, "test.py")
        process = subprocess.Popen(
            ["python", model_file],
            cwd=model_folder,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait until Gradio prints the local URL which includes the port.
        port = None
        start_time = time.time()
        while time.time() - start_time < 30:  # wait max 30 seconds
            output_line = process.stdout.readline()
            if output_line:
                match = re.search(r"http://127\.0\.0\.1:(\d+)", output_line)
                if match:
                    port = int(match.group(1))
                    break
            time.sleep(0.1)
        
        if port is None:
            port = 7860  # fallback if port cannot be determined
            gradio_servers[model_name] = {"port": port, "process": process}
        else:
            port = gradio_servers[model_name]["port"]

    # Return a wrapper HTML page hosting the Gradio interface in an iframe.
    iframe_page = f"""
    <html>
    <head>
      <title>Model Interface - {model_name}</title>
    </head>
    <body>
      <h1>{model_name} Interface</h1>
      <iframe src="http://localhost:{port}" width="100%" height="600px" frameBorder="0"></iframe>
      <p><a href="/models">Back to Models List</a></p>
    </body>
    </html>
    """
    return iframe_page

if __name__ == "__main__":
    app.run(debug=True, port=5000)