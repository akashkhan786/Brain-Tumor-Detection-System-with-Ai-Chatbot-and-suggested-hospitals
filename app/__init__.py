from flask import Flask
from app.routes import setup_routes

app = Flask(__name__,template_folder='../templates')
setup_routes(app)
