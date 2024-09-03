from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from config import Config

db = SQLAlchemy()
socketio = SocketIO()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    
    # Initialize the database
    db.init_app(app)

    # Initialize Flask-SocketIO
    socketio.init_app(app)
    
    # Import and register the blueprint
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    # Ensure models are imported and tables are created
    with app.app_context():
        from app import models
        db.create_all()

    return app
