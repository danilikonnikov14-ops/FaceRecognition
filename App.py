"""
Веб-сервис распознавания лиц.
Этот модуль предоставляет веб-интерфейс для загрузки фотографий и
распознавания лиц с использованием нейросетевых моделей.

"""

import os
import io
import warnings

import numpy as np
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
mtcnn = None
face_database = {}
class_names = []
embeddings = {}

model = "models/face_recognition_model.pth"

def load_model():
    global model, mtcnn, face_database, class_names, embeddings
    
    print(f"Загрузка модели на устройство: {device}")
    
    try:
        checkpoint = torch.load(model, map_location=device, weights_only=False)
    except:
        checkpoint = torch.load(model, map_location=device)
    
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=device
    )
    
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
    
    if 'embeddings' in checkpoint:
        embeddings = checkpoint['embeddings']
    
    face_database = {}
    for name, emb in embeddings.items():
        if isinstance(emb, np.ndarray):
            face_database[name] = torch.tensor(emb).to(device)
        else:
            face_database[name] = emb.to(device) if torch.is_tensor(emb) else torch.tensor(emb).to(device)
    
    print(f"Модель загружена. Найдено {len(class_names)} классов")
    print(f"Доступные классы: {class_names}")

def cosine_similarity(emb1, emb2):
    """Вычисление косинусного сходства между эмбеддингами"""
    return torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

def recognize_face(face_embedding, threshold=0.5, recognized_names=None):
    """Распознавание лица по эмбеддингу с учетом уже распознанных имен"""
    if not face_database:
        return "Неизвестный", 0.0
    
    if recognized_names is None:
        recognized_names = set()
    
    best_match = "Неизвестный"
    best_similarity = -1
    
    for name, db_embedding in face_database.items():
        if name in recognized_names:
            continue
            
        similarity = cosine_similarity(face_embedding, db_embedding)
        if similarity > best_similarity and similarity > threshold:
            best_similarity = similarity
            best_match = name
    
    return best_match, best_similarity

def process_face(face_img):
    """Обработка одного лица и получение эмбеддинга"""
    try:
        face = mtcnn(face_img)
        
        if face is not None:
            face = face.unsqueeze(0).to(device)
            
            with torch.no_grad():
                face_embedding = model(face).squeeze()
            
            return face_embedding
        else:
            return None
    except Exception as e:
        print(f"Ошибка при обработке лица: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Нет выбранного файла'}), 400
    
    if file:
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert('RGB')
            
            boxes, _ = mtcnn.detect(img)
            
            if boxes is None or len(boxes) == 0:
                return jsonify({'faces': []})
            
            recognized_names = set()
            
            faces_data = []
            for i, box in enumerate(boxes):
                try:
                    box = [int(coord) for coord in box]
                    x1, y1, x2, y2 = box
                    
                    padding = 10
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(img.width, x2 + padding)
                    y2 = min(img.height, y2 + padding)
                    
                    face_img = img.crop((x1, y1, x2, y2))
                    
                    face_embedding = process_face(face_img)
                    
                    if face_embedding is not None:
                        name, confidence = recognize_face(face_embedding, recognized_names=recognized_names)
                        
                        if name != "Неизвестный":
                            recognized_names.add(name)
                        
                        faces_data.append({
                            'id': i,
                            'box': [float(x) for x in box], 
                            'name': name,
                            'confidence': float(confidence)
                        })
                    else:
                        faces_data.append({
                            'id': i,
                            'box': [float(x) for x in box],
                            'name': "Не удалось обработать",
                            'confidence': 0.0
                        })
                        
                except Exception as e:
                    print(f"Ошибка обработки лица {i}: {e}")
                    faces_data.append({
                        'id': i,
                        'box': [float(x) for x in box] if 'box' in locals() else [0, 0, 0, 0],
                        'name': "Ошибка",
                        'confidence': 0.0
                    })
            
            return jsonify({'faces': faces_data})
            
        except Exception as e:
            print(f"Общая ошибка: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    if os.path.exists(model):
        try:
            load_model()
            print("Модель успешно загружена!")
            app.run(debug=True, host='0.0.0.0', port=5000)
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            print("Пожалуйста, переобучите модель с помощью train.py")
    else:
        print("Ошибка: Модель не найдена. Сначала запустите train.py")