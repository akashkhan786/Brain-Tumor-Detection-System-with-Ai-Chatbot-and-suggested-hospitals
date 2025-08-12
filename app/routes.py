import os
from flask import request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename, safe_join
from app.image_model import predict
from app.chatbot import ask_question
from app.config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def setup_routes(app):
    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            file = request.files['file']
            if file and allowed_file(file.filename):
                file_location = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
                file.save(file_location)
                result, confidence = predict(file_location)
                return render_template('index.html', result=result, confidence=f'{confidence*100:.2f}%', file_path=f'/uploads/{file.filename}')
        return render_template('index.html', result=None)

    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        safe_path = safe_join(UPLOAD_FOLDER, filename)
        if os.path.exists(safe_path):
            return send_from_directory(UPLOAD_FOLDER, filename)
        else:
            return f"File not found: {filename}", 404

    @app.route('/ask', methods=['POST'])
    def ask():
        user_query = request.json.get('query')
        response = ask_question(user_query)
        return jsonify(response)
