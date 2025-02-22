from flask import Flask, render_template, request, jsonify, send_file
import os
import PyPDF2
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from reportlab.pdfgen import canvas
import json
from werkzeug.utils import secure_filename
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def generate_flashcards(text):
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    prompt = f"""Given the following text, generate up to 10 high-quality flashcards. 
    Each flashcard should have a clear question on the front and a concise answer on the back. 
    Focus on the most important concepts and key information.
    
    Text: {text}
    
    Format each flashcard as a JSON object with 'id', 'front', and 'back' fields. 
    Return only the JSON array of flashcards.
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract the flashcards from the response
        flashcards_text = response.choices[0].message.content
        # Clean up the response to ensure it's valid JSON
        flashcards_text = flashcards_text.strip()
        if flashcards_text.startswith('```json'):
            flashcards_text = flashcards_text[7:]
        if flashcards_text.endswith('```'):
            flashcards_text = flashcards_text[:-3]
        
        flashcards = json.loads(flashcards_text)
        return flashcards
    except Exception as e:
        print(f"Error generating flashcards: {str(e)}")
        # Fallback to basic flashcard generation if AI fails
        sentences = sent_tokenize(text)
        flashcards = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) > 5:
                flashcards.append({
                    'id': i,
                    'front': sentence,
                    'back': ' '.join(sentence.split()[len(sentence.split())//2:])
                })
        
        return flashcards[:min(len(flashcards), 10)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            text = extract_text_from_pdf(filepath)
            flashcards = generate_flashcards(text)
            
            # Save flashcards to a JSON file
            cards_filename = f"{filename}_cards.json"
            cards_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cards_filename)
            with open(cards_filepath, 'w') as f:
                json.dump(flashcards, f)
            
            return jsonify({
                'message': 'File successfully processed',
                'flashcards': flashcards
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/export-pdf', methods=['POST'])
def export_pdf():
    flashcards = request.json.get('flashcards')
    if not flashcards:
        return jsonify({'error': 'No flashcards provided'}), 400
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'flashcards.pdf')
    c = canvas.Canvas(output_path)
    
    # Set up PDF formatting
    margin = 50
    page_width = 550
    page_height = 750
    
    for i, card in enumerate(flashcards):
        if i > 0 and i % 4 == 0:
            c.showPage()
        
        y_position = page_height - (i % 4) * 180 - margin
        
        # Draw card
        c.rect(margin, y_position - 150, page_width - 2*margin, 140)
        
        # Add content
        c.setFont("Helvetica", 12)
        c.drawString(margin + 10, y_position - 30, f"Front: {card['front'][:100]}...")
        c.drawString(margin + 10, y_position - 90, f"Back: {card['back'][:100]}...")
    
    c.save()
    return send_file(output_path, as_attachment=True, download_name='flashcards.pdf')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
