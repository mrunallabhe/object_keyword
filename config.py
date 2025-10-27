"""
Configuration file for the OCR & Object Detection System
"""

import os

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB max file size
    
    # File paths
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER') or 'outputs'
    
    # Database
    DATABASE_PATH = 'ocr_database.db'
    
    # OCR settings
    OCR_LANGUAGES = ['en']  # Add more languages as needed: ['en', 'es', 'fr']
    OCR_GPU = True  # Set to False to force CPU usage
    
    # Object detection settings
    DETECTION_BOX_THRESHOLD = 0.25
    DETECTION_TEXT_THRESHOLD = 0.25
    
    # CLIP settings (fallback detection)
    CLIP_MODEL_NAME = "ViT-B/32"
    CLIP_SIMILARITY_THRESHOLD = 0.22
    CLIP_NMS_THRESHOLD = 0.35
    
    # Image processing settings
    MAX_IMAGE_SIZE = 2000  # Resize images larger than this
    DESKEW_ENABLED = True
    LINE_REMOVAL_ENABLED = True
    
    # License plate detection
    PLATE_MIN_LENGTH = 5
    PLATE_MAX_LENGTH = 12
    
    # PDF generation
    PDF_PAGE_SIZE = 'A4'
    PDF_IMAGE_WIDTH = 400
    PDF_IMAGE_HEIGHT = 300

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # In production, set these via environment variables:
    # SECRET_KEY = os.environ.get('SECRET_KEY')
    # UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER')
    # OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER')

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    UPLOAD_FOLDER = 'test_uploads'
    OUTPUT_FOLDER = 'test_outputs'
    DATABASE_PATH = 'test_ocr_database.db'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
