from flask import Flask, request, jsonify
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import os

app = Flask(__name__)

# Initialize FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=0)

# Load known embeddings (you can make this dynamic later)
known_embeddings = {}  # {'name': embedding}

# Example: load registered face embeddings
def load_known_faces():
    global known_embeddings
    for filename in os.listdir('known_faces'):
        if filename.endswith('.npy'):
            name = filename.split('.')[0]
            embedding = np.load(f'known_faces/{filename}')
            known_embeddings[name] = embedding
load_known_faces()

# Calculate Euclidean distance
def calculate_distance(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)

# Face recognition endpoint
@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Detect face and get embedding
    face = mtcnn(img)
    if face is None:
        return jsonify({'error': 'No face detected'}), 400

    face = face.unsqueeze(0).to(device)
    embedding = model(face).detach().cpu().numpy()

    # Compare with known faces
    min_distance = float('inf')
    identity = "Unknown"

    for name, known_emb in known_embeddings.items():
        distance = calculate_distance(embedding, known_emb)
        if distance < min_distance:
            min_distance = distance
            identity = name

    if min_distance > 0.9:  # Adjust threshold based on testing
        identity = "Unknown"

    return jsonify({'name': identity, 'distance': float(min_distance)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
