# import io
# import sys
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# # Define the model architecture (same as used in training)
# class MNISTModel(nn.Module):
#     def __init__(self):
#         super(MNISTModel, self).__init__()
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc_layer = nn.Sequential(
#             nn.Linear(64 * 7 * 7, 128),
#             nn.ReLU(),
#             nn.Linear(128, 10)
#         )

#     def forward(self, x):
#         x = self.conv_layer(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layer(x)
#         return x

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Load the trained model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MNISTModel().to(device)
# model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
# model.eval()

# # Define the transform (should match training)
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# # Return an inline HTML page that lets users upload an image.
# @app.route("/")
# def index():
#     html_content = """
#     <html>
#     <head><title>MNIST Model Interface</title></head>
#     <body>
#       <h1>MNIST Model Interface</h1>
#       <form action="/predict" method="post" enctype="multipart/form-data">
#          <label>Upload Image:</label>
#          <input type="file" name="file" accept="image/*"><br><br>
#          <input type="submit" value="Predict">
#       </form>
#     </body>
#     </html>
#     """
#     return html_content

# # Prediction endpoint
# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file provided"}), 400

#     file = request.files["file"]
#     try:
#         img_bytes = file.read()
#         image = Image.open(io.BytesIO(img_bytes)).convert("L")
#         image = transform(image).unsqueeze(0).to(device)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs, 1)

#     return jsonify({"prediction": predicted.item()})

# # Accept a port argument so it can be launched by a parent process.
# if __name__ == "__main__":
#     port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
#     app.run(debug=True, port=port)
import gradio as gr

def greeting(name):
    return f"Hello {name} !"

interface = gr.Interface(
    fn=greeting,
    inputs="text",
    outputs="text",
    title="Greeting App",
    description="Enter your name to receive a greeting.",
)

if __name__ == "__main__":
    interface.launch()