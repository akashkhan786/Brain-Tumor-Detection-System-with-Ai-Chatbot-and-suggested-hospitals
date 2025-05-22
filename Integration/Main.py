# ----------------------- Import Dependencies -----------------------
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename, safe_join
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd
import os
import re
from fuzzywuzzy import fuzz, process

# -------------------------- Flask Setup --------------------------
app = Flask(__name__)
load_dotenv(find_dotenv())

# -------------------- Image Model Configuration --------------------
model = load_model('../Resnet_model/model/best_model.keras', compile=False)
class_labels = ['glioma', 'notumor','meningioma', 'pituitary']

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'Resnet_model', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# ---------------------- Chatbot Configuration ----------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

DB_FAISS_PATH = "../Chatbot/vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# -------------------------- Utility Functions --------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction)
    return class_labels[predicted_class[0]], confidence

# -------------------------- Enhanced Hospital Chatbot Logic --------------------------
def load_hospital_data():
    """Load and clean hospital data"""
    file_path = '../Chatbot/data/hospital_data.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Clean data
        df = df.dropna(how='all')
        df = df.fillna('Not Available')
        # Standardize columns
        df.columns = [col.strip().upper() for col in df.columns]
        return df
    return pd.DataFrame()

class HospitalChatbot:
    def __init__(self, df):
        self.df = df
        self.current_results = pd.DataFrame()
        self.current_page = 0
        self.page_size = 5
        
        # Attribute mapping dictionary
        self.attribute_map = {
            'CONTACT': ['contact', 'phone', 'number', 'mobile', 'phone no', 'mobile no'],
            'ADDRESS': ['address', 'location', 'where is', 'located', 'place'],
            'DOCTORS': ['doctor', 'doctors', 'specialist'],
            'SERVICES': ['service', 'facility', 'facilities'],
            'HOSPITAL NAME': ['name', 'hospital']
        }

    def map_query_to_column(self, query):
        query = query.lower()
        for col, keywords in self.attribute_map.items():
            if any(keyword in query for keyword in keywords):
                return col
        return None

    def fuzzy_search_hospitals(self, query, city=None):
        query = query.lower()
        
        # Filter by city if mentioned
        df = self.df
        if city:
            df = df[df['CITY'].str.lower() == city.lower()]
        
        # Search in hospital names with fuzzy matching
        hospitals = df['HOSPITAL NAME'].tolist()
        matches = process.extract(query, hospitals, scorer=fuzz.partial_ratio, limit=10)
        
        # Get matches with score > 70
        matched_hospitals = [match[0] for match in matches if match[1] > 70]
        
        if matched_hospitals:
            return df[df['HOSPITAL NAME'].isin(matched_hospitals)]
        
        # Fallback to searching in all fields
        return df[df.apply(lambda row: any(query in str(row[col]).lower() 
                          for col in ['HOSPITAL NAME', 'ADDRESS', 'CITY']), axis=1)]

    def get_next_page(self):
        start = self.current_page * self.page_size
        end = start + self.page_size
        self.current_page += 1
        
        if start >= len(self.current_results):
            return None
        
        return self.current_results.iloc[start:end]

    def format_response(self, hospitals, specific_info=None):
        if hospitals.empty:
            return "No hospitals found."
            
        responses = []
        for _, row in hospitals.iterrows():
            response = []
            if specific_info:
                if specific_info in row:
                    response.append(f"{specific_info.title()}: {row[specific_info]}")
                else:
                    return f"{specific_info} information not available for this hospital."
            else:
                response.append(f"🏥 {row['HOSPITAL NAME']}")
                for col in ['CITY', 'ADDRESS', 'CONTACT', 'DOCTORS']:
                    if col in row and row[col] != 'Not Available':
                        response.append(f"📍 {col}: {row[col]}" if col == 'ADDRESS' else f"📞 {col}: {row[col]}")
            
            responses.append("\n".join(response))
        
        return "\n\n".join(responses)

    def handle_query(self, query):
        # Handle basic greetings
        if re.search(r'\b(hi|hello|hey|whatsup|what\'s up|salam|assalamualaikum)\b', query.lower()):
            return {
                "result": "Hi! How can I help you today?",
                "source": "System"
            }

        # Check if asking for more results
        if any(word in query.lower() for word in ['more', 'next', 'further']):
            next_page = self.get_next_page()
            if next_page is not None:
                return {
                    "result": self.format_response(next_page),
                    "source": "Hospital database"
                }
            return {
                "result": "No more hospitals available.",
                "source": "System"
            }
        
        # Check if asking for specific information
        column = self.map_query_to_column(query)
        
        # Extract hospital name if mentioned
        hospital_name = None
        for name in self.df['HOSPITAL NAME']:
            if name.lower() in query.lower():
                hospital_name = name
                break
        
        # If specific hospital mentioned
        if hospital_name:
            hospital_data = self.df[self.df['HOSPITAL NAME'] == hospital_name]
            if not hospital_data.empty:
                if column:
                    return {
                        "result": self.format_response(hospital_data, column),
                        "source": "Hospital database"
                    }
                return {
                    "result": self.format_response(hospital_data),
                    "source": "Hospital database"
                }
            return {
                "result": f"Hospital '{hospital_name}' not found.",
                "source": "System"
            }
        
        # Search for hospitals in city
        city = None
        for known_city in self.df['CITY'].unique():
            if known_city.lower() in query.lower():
                city = known_city
                break
        
        # Perform search
        self.current_results = self.fuzzy_search_hospitals(query, city)
        self.current_page = 0
        
        first_page = self.get_next_page()
        if first_page is not None:
            response = {
                "result": self.format_response(first_page),
                "source": "Hospital database"
            }
            if len(self.current_results) > self.page_size:
                response["result"] += "\n\nSay 'more' to see additional results."
            return response
        
        return {
            "result": "No hospitals found matching your query.",
            "source": "System"
        }

# Initialize hospital data and chatbot
df_hospitals = load_hospital_data()
hospital_chatbot = HospitalChatbot(df_hospitals)

# -------------------------- Routes --------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_location = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_location)
            result, confidence = predict(file_location)
            return render_template('index.html', result=result, confidence=f'{confidence*100:.2f}%', file_path=f'/uploads/{filename}')
    return render_template('index.html', result=None)

@app.route('/chatbot')
def chatbot_page():
    return render_template('pages/chatbot.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_query = request.json.get('query')
        page = int(request.json.get('page', 1))
        
        # Handle the query using our enhanced chatbot
        response = hospital_chatbot.handle_query(user_query)
        
        # If no results from direct search, fallback to RAG
        if "No hospitals found" in response["result"]:
            rag_response = qa_chain.invoke({'query': user_query})
            return jsonify({
                "result": rag_response.get("result"),
                "source_documents": [doc.page_content for doc in rag_response.get("source_documents", [])]
            })
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Error in /ask endpoint: {str(e)}")
        return jsonify({
            "result": "Sorry, an error occurred while processing your request.",
            "error": str(e)
        }), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    safe_path = safe_join(UPLOAD_FOLDER, filename)
    if os.path.exists(safe_path):
        return send_from_directory(UPLOAD_FOLDER, filename)
    else:
        return f"File not found: {filename}", 404

# -------------------------- Run App --------------------------
if __name__ == '__main__':
    # Verify data loaded correctly
    if df_hospitals.empty:
        print("Warning: Hospital data failed to load or is empty!")
    else:
        print(f"Hospital data loaded successfully with {len(df_hospitals)} records")
    
    app.run(debug=True, use_reloader=False)
