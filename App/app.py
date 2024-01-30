from flask import Flask, render_template, request, redirect, url_for, flash, send_file, send_from_directory, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util
import numpy as np
import cv2
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import fitz
import os
from pymongo import MongoClient
import gridfs
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/MdgProject'

client = MongoClient('mongodb://localhost:27017/')
db = client['MdgProject']
fs = gridfs.GridFS(db)
mongo = PyMongo(app)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

use_model = SentenceTransformer('paraphrase-mpnet-base-v2')

class User:
    def __init__(self, username, password):
        self.username = username
        self.password = password


def get_use_embeddings(lines):
    return use_model.encode(lines, convert_to_tensor=True)

def find_similar_lines_bert(lines, user_query):
    user_query_embedding = get_use_embeddings([user_query])[0]
    page_lines_embeddings = get_use_embeddings(lines)

    similarities = util.pytorch_cos_sim(user_query_embedding, page_lines_embeddings)[0]

    threshold = 0.6
    similar_indices = [i for i, sim in enumerate(similarities) if sim.item() > threshold]

    similar_lines = [(lines[i], similarities[i].item()) for i in similar_indices]
    return similar_lines

def pre_proc(im):
    img = np.array(im)

    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = grayscale(img)
    thresh, im_bw = cv2.threshold(gray_image, 130, 200, cv2.THRESH_BINARY)
    return im_bw


def perform_ocr_image(image_path):
    img = Image.open(image_path)

    img = pre_proc(img)

    ocr_text = pytesseract.image_to_string(img)

    return ocr_text


def perform_ocr_pdf(pdf_path, page_number):
    doc = fitz.open(pdf_path)

    page = doc[page_number]

    image = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    img = Image.frombytes("RGB", [image.width, image.height], image.samples)

    img = pre_proc(img)

    ocr_text = pytesseract.image_to_string(img)

    doc.close()

    return ocr_text


def get_pdf_len(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count
        doc.close()

        return num_pages
    except Exception as e:
        print(f"Error: {e}")
        return None


def give_ocr_array(pdf_path):
    pdf_pages = get_pdf_len(pdf_path)
    e_text = []
    for i in range(pdf_pages):
        if (i + 1) % 10 == 0:
            print(i + 1)
        text = perform_ocr_pdf(pdf_path, i)
        text = text.split('\n')
        e_text.append((i + 1, text))
    return e_text


def save_text_to_database(filename, text_array):
    try:
        mongo.db.books.insert_one(
            {'filename': filename, 'text_array': text_array})
        print("Text array saved to database")
    except Exception as e:
        print("Error saving text array to database:", str(e))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if mongo.db.users.find_one({'username': username}):
            flash('Username already exists. Choose a different one.', 'error')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(
            password, method='pbkdf2:sha256')

        mongo.db.users.insert_one(
            {'username': username, 'password': hashed_password})

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = mongo.db.users.find_one({'username': username})
        if user and check_password_hash(user['password'],  password):
            flash('Login successful!', 'success')
            session['username'] = username
            return redirect(url_for('dashboard'))

        flash('Invalid username or password. Please try again.', 'error')

    return render_template('login.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    username = session.get('username')
    if request.method == 'POST':
        option = request.form['option']
        if option == 'Library':
            return redirect(url_for('library'))
        elif option == 'Search':
            books = mongo.db.books.find()
            return render_template('index.html', books=books)

    return render_template('dashboard.html', current_user=username)


class Book:
    def __init__(self, title, content):
        self.title = title
        self.content = content


ALLOWED_EXTENSIONS = {'pdf'}

UPLOAD_FOLDER = r'C:\Users\gargn\Downloads\SOCCC\uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/library', methods=['GET', 'POST'])
def library():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                text_array = give_ocr_array(os.path.join(
                    app.config['UPLOAD_FOLDER'], filename))
                save_text_to_database(filename, text_array)
                return redirect(url_for('library'))

    existing_books = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('library.html', books=existing_books)
    

@app.route('/view_book/<filename>')
def view_book(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    user_query = request.form['user_query']
    uploaded_file = request.files['file']
    filename = request.form['filename']
    if uploaded_file.filename != '':
        file_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(file_path)

        image_text = perform_ocr_image(file_path)
        user_query = image_text

    book = mongo.db.books.find_one({'filename': filename})
    e_text = book['text_array']
    similar_lines = []
    count_dict = {}
    for i, lines in e_text:
        sim = find_similar_lines_bert(lines, user_query)
        if sim:
            for a, b in sim:
                similar_lines.append((i, a, b))
                if i in count_dict:
                    count_dict[i] += 1
                else:
                    count_dict[i] = 1
    similar_lines = sorted(similar_lines, key=lambda x: x[2], reverse=True)
    if similar_lines:
        max_page = max(count_dict, key=count_dict.get)
        return render_template('results.html', user_query=user_query, similar_lines=similar_lines, max_page=max_page)
    else:
        return render_template('results.html', user_query=user_query, no_results=True)



if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
