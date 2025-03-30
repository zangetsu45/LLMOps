from flask import Flask, request, redirect, url_for, Response, render_template
import os
import threading
from werkzeug.utils import secure_filename
import gradio as gr
import subprocess
import socket
import re, time
import requests

app = Flask(__name__)
UPLOAD_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Home page: links to list models and model upload form
@app.route("/")
def index():
    return render_template("index.html")

# Endpoint to handle model uploads
@app.route("/upload", methods=["POST"])
def upload_model():
    required_files = ["model_definition", "model_weights", "requirements"]
    for file_key in required_files:
        if file_key not in request.files:
            return f"No file provided for {file_key}", 400

    # Get model_name from form data
    model_name = request.form.get("model_name", "").strip()
    if not model_name:
        return "Model name is required", 400

    model_def_file = request.files["model_definition"]
    weights_file = request.files["model_weights"]
    req_file = request.files["requirements"]

    if model_def_file.filename == "" or weights_file.filename == "" or req_file.filename == "":
        return "One or more files were not selected", 400

    # Use the provided model_name (force sanitized) as the model identifier.
    model_folder = os.path.join(UPLOAD_FOLDER, secure_filename(model_name))
    os.makedirs(model_folder, exist_ok=True)

    # Force saving the model definition as app.py and requirements as requirements.txt.
    model_def_path = os.path.join(model_folder, "app.py")
    weights_path = os.path.join(model_folder, secure_filename(weights_file.filename))
    req_path = os.path.join(model_folder, "requirements.txt")

    model_def_file.save(model_def_path)
    weights_file.save(weights_path)
    req_file.save(req_path)

    return redirect(url_for("list_models"))

# Endpoint to list available models
@app.route("/models", methods=["GET"])
def list_models():
    models = os.listdir(UPLOAD_FOLDER)
    return render_template("models.html", models=models)

# Dictionary to store launched gradio interfaces and their ports.
gradio_servers = {}

@app.route("/model/<model_name>", methods=["GET"])
def model_specific(model_name):
    # Check if the model exists in our models directory.
    if (model_name not in os.listdir(UPLOAD_FOLDER)):
        return "Model not found", 404

    # If this model's server hasn't been started yet, create and launch it.
    if model_name not in gradio_servers:
        model_folder = os.path.join(UPLOAD_FOLDER, model_name)
        # Launch the model process without forcing a port.
        # Since we cannot modify test.py, we capture its stdout to parse the port at runtime.
        model_file = os.path.join(model_folder, "app.py")
        print(f"Launching model server for {model_name}...")
        print(f"Model file: {model_file}")

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        env = os.environ.copy()
        env["GRADIO_SERVER_PORT"] = str(port)
        env["GRADIO_ROOT_PATH"] = f"/model/{model_name}"
        process = subprocess.Popen(
            ["python", "app.py"],
            cwd=model_folder,
            env=env,
        )
        gradio_servers[model_name] = {"process": process, "port": port}
    else:
        port = gradio_servers[model_name]["port"]
    # Return a template hosting the Gradio interface in an iframe.
    return render_template("model_interface.html", model_name=model_name, port=port)


# Endpoint to handle model API requests does not get called in frontend
@app.route("/model/<model_name>/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def proxy_model_api(model_name, subpath):
    # Ensure the model's Gradio container is running
    if model_name not in gradio_servers:
        return f"Model {model_name} is not running", 404

    port = gradio_servers[model_name]["port"]

    # Build the target URL (note the leading '/' for subpath) and include query parameters
    target_url = f"http://localhost:{port}/{subpath}"
    params = dict(request.args)
    if "session_hash" not in params:
        params["session_hash"] = "1234"
    print(f"Proxying request for {model_name} to {target_url} with params {params}")

    # Forward the original request (method, headers, data, etc)
    resp = requests.request(
        method=request.method,
        url=target_url,
        params=params,
        headers={key: value for key, value in request.headers if key != "Host"},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False
    )
    excluded_headers = ["content-encoding", "content-length", "transfer-encoding", "connection"]
    headers = [(name, value) for name, value in resp.raw.headers.items() if name.lower() not in excluded_headers]

    return Response(resp.content, resp.status_code, headers)

# Add new routes for API Doc and Instances Running under a specific model
@app.route("/model/<model_name>/api_doc")
def api_doc_model(model_name):
    if model_name not in os.listdir(UPLOAD_FOLDER):
        return "Model not found", 404
    return render_template("api_doc.html", model_name=model_name)

@app.route("/model/<model_name>/instances")
def instances_model(model_name):
    count = 1 if model_name in gradio_servers else 0
    return render_template("instances.html", model_name=model_name, count=count)

if __name__ == "__main__":
    app.run(debug=True, port=5000)