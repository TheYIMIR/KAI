# setup_nltk.py
import nltk

def download_nltk_data():
    """Download required NLTK data"""
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')

if __name__ == "__main__":
    download_nltk_data()