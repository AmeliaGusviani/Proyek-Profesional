from flask import Flask, request, render_template, url_for, jsonify
import os
import cv2
import numpy as np
import pickle
from skimage.feature import hog
from skimage.color import rgb2gray
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask import session, redirect, flash
from geopy.geocoders import Nominatim
import requests
from datetime import datetime
import pytz
from sqlalchemy import asc


import os
app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db_user = "ameliagusviani"
db_password = "mysqladmin"  # ganti dengan password MySQL kamu
db_host = "ameliagusviani.mysql.pythonanywhere-services.com"
db_name = "ameliagusviani$cassava"
db_port = "3306"

app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

db = SQLAlchemy(app)

with app.app_context():
    db.create_all()

# Cek file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load models
MODEL_DIR = os.path.join(BASE_DIR, 'models')

with open(os.path.join(MODEL_DIR, 'svm_hog_pca_smote_model.pkl'), 'rb') as f:
    svm_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'pca_model_old.pkl'), 'rb') as f:
    pca_model = pickle.load(f)


# Load CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    os.path.join(MODEL_DIR, 'best_model.pth'),
    map_location=device
)

cnn_model = efficientnet_b0(weights=None)
num_ftrs = cnn_model.classifier[1].in_features
cnn_model.classifier[1] = nn.Linear(num_ftrs, len(checkpoint["classes"]))
cnn_model.load_state_dict(checkpoint["model_state"])
cnn_model.to(device)
cnn_model.eval()

cnn_classes = checkpoint["classes"]

# Info kelas penyakit
class_map = {
    0: 'CBB',
    1: 'CBSD',
    2: 'CGM',
    3: 'CMD',
    4: 'Healthy'
}

# Informasi Penyakit saat Klasifikasi
disease_info_map = {
    'CBB': (
        """Penyebab dan Pola Infeksi
    Cassava Bacterial Blight (CBB) adalah penyakit pada tanaman singkong yang disebabkan oleh bakteri Xanthomonas axonopodis pv manihotis.
    Penyakit ini sering muncul di daerah dengan kelembapan tinggi atau saat musim hujan.
    Gejala Spesifik
    Gejalanya meliputi bintik-bintik hitam pada daun dan bercak layu (blight), daun yang terinfeksi akan mengering sebelum waktunya dan rontok akibat layu. 
    Selain itu, biasanya muncul cairan lengket seperti getah di bagian bawah daun, tangkai daun, dan batang, terutama saat udara lembap atau musim hujan.
    Penanganan dan Pengendalian
    Penanganan yang dapat diterapkan meliputi rotasi tanaman, penggunaan varietas tahan penyakit, serta sanitasi dan pengendalian kelembapan lahan. 
    - Cabut dan bakar tanaman yang terinfeksi untuk memutus sumber inokulum.
    - Gunakan stek atau tanaman bebas bakteri sebagai sumber perbanyakan.
    - Hindari menanam singkong berturut-turut di lahan yang sama dalam beberapa musim.
    - Pemangkasan agar sirkulasi udara baik, penjarangan tanaman agar cahaya & ventilasi cukup."""
    ),

    'CBSD': (
        """Penyebab dan Pola Infeksi
    Cassava Brown Streak Disease (CBSD) adalah penyakit yang disebabkan oleh dua jenis virus, yaitu Cassava brown streak virus (CBSV) dan Ugandan Cassava Brown Streak Virus (UCBSV). 
    Virus ini menyebar melalui vektor putih (Bemisia tabaci) serta melalui stek tanaman yang terinfeksi. 
    Gejala Spesifik
    Gejala khas CBSD meliputi perubahan warna menjadi kuning pada tulang daun, yang terkadang melebar membentuk bercak kuning besar. 
    CBSD juga ditandai dengan adanya area berwarna cokelat gelap pada umbi singkong, disertai dengan penurunan ukuran umbi. 
    Penanganan dan Pengendalian
    Penanganan yang dapat diterapkan meliputi penggunaan varietas tanaman dan stek sehat (bebas virus). 
    - Stek bebas virus dengan uji diagnostik (misalnya RT-PCR) sebelum ditanam.
    - Menghapus tanaman sakit secara massal dalam komunitas untuk mengurangi inokulum virus. 
    - Pengendalian populasi kutu putih melalui teknik IPM (serangga pemangsa, perangkap kuning, insektisida selektif bila perlu)
    - Pengembangan tanaman transgenik atau sistem RNA interference untuk ketahanan virus sedang dieksplorasi."""
    ),

    'CGM': (
        """Penyebab dan Pola Infeksi
    Cassava Green Mite (CGM) adalah gangguan yang disebabkan oleh tungau Mononychellus tanajoa, yang mengisap cairan daun muda dan menyebabkan daun menjadi cerah, kecil, dan belang-belang. 
    Serangan CGM menyebabkan munculnya bintik-bintik kecil, yang lama-kelamaan menyebar dan secara bertahap membesar menutupi seluruh permukaan daun. 
    Gejala Spesifik
    Daun yang terinfeksi akan mengering, menyusut, dan akhirnya rontok dari tanaman. 
    Pada kasus yang parah, CGM juga memunculkan gejala belang-belang (mottling) yang disalahartikan sebagai gejala penyakit Cassava Mosaic Disease (CMD). 
    Penanganan dan Pengendalian
    Penanganan yang dapat diterapkan meliputi pengendalian hayati dengan memanfaatkan musuh alami seperti tungau predator, penggunaan varietas tanaman, penerapan sanitasi lahan, serta rotasi tanaman. 
    - Memanfaatkan predator alami seperti tungau pemangsa (misalnya Phytoseiulus, Neoseiulus), atau nematoda entomopatogen jika sesuai.
    - Menyemprot daun pada pagi hari agar tungau terjatuh dan kelembapan naik.
    - Membuang daun yang sangat terinfeksi agar populasi tungau tidak meningkat.
    - Pengaturan jarak dan kepadatan tanamana agar sirkulasi udara baik dan mengurangi stres tanaman."""
    ),

    'CMD': (
        """Penyebab dan Pola Infeksi
    Cassava Mosaic Disease (CMD) merupakan penyakit pada tanaman singkong yang disebabkan oleh virus Geminivirus dan ditularkan oleh serangga vektor kutu putih (Bemisia tabaci). 
    Penyakit ini menunjukkan berbagai gejala pada daun, seperti belang-belang (mottling), mosaik karat, daun yang melintir, serta penyusutan ukuran daun dan tanaman secara keseluruhan. 
    Gejala Spesifik
    Daun yang terinfeksi biasanya memiliki bercak hijau yang bercampur dengan warna kuning dan putih. 
    Bercak ini mengurangi luas permukaan daun yang berperan penting dalam proses fotosintesis, sehingga menyebabkan pertumbuhan tanaman terhambat dan hasil panen berkurang. 
    Penanganan dan Pengendalian
    Penanganan yang dapat diterapkan meliputi penggunaan varietas singkong tahan penyakit, penerapan bahan tanam sehat bebas penyakit, pengendalian vektor baik melalui pendekatan hayati maupun kimia, serta tindakan sanitasi seperti pencabutan tanaman terinfeksi. 
    - pengurangan populasi kutu putih melalui agen hayati, insektisida selektif, kontrol lingkungan (misalnya tanaman penarik).
    - mencabut tanaman yang terinfeksi parah agar tidak menjadi sumber virus bagi tanaman lain.
    - menanam tanaman lain sebagai jeda untuk memutus siklus penyakit.
    - Penggunaan varietas yang memiliki resistensi terhadap virus CMD banyak dikembangkan."""
    ),

    'HEALTHY': (
        "Daun singkong dalam kondisi sehat dan tidak menunjukkan gejala penyakit apa pun. "
        "Permukaannya berwarna hijau merata, tidak terdapat bintik, bercak, atau pola warna abnormal. "
        "Tanaman sehat akan tumbuh optimal dan memberikan hasil panen maksimal. "
        "Tetap jaga kesehatan tanaman dengan pemupukan seimbang, pengendalian hama, serta penggunaan benih yang berkualitas. "
    )
}


def get_location_name(lat, lon):
    geolocator = Nominatim(user_agent="cassava_app")
    try:
        location = geolocator.reverse((lat, lon), language='id')
        return location.address.split(",")[2:4]  # misalnya ambil kota dan provinsi
    except:
        return "Tidak diketahui"

def get_city_from_coords(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        response = requests.get(url, headers={"User-Agent": "cassava-classification"})
        data = response.json()
        city = data.get("address", {}).get("city") or \
               data.get("address", {}).get("town") or \
               data.get("address", {}).get("village") or \
               data.get("address", {}).get("state")
        return city or "Tidak diketahui"
    except Exception:
        return "Tidak diketahui"
    
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)  # ganti password_hash → password
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    histories = db.relationship('ClassificationHistory', backref='user', lazy=True)

class ClassificationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(pytz.timezone('Asia/Jakarta')))
    location = db.Column(db.String(100), nullable=True)
    method = db.Column(db.String(50), nullable=True) 

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        if User.query.filter_by(username=username).first():
            flash('Username sudah digunakan')
            return redirect('/register')
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        flash('Registrasi berhasil. Silakan login.')
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            session['username'] = user.username  
            flash('Login berhasil')
            return redirect('/')
        else:
            flash('Username atau password salah')
            return render_template('login.html', username=username)
    return render_template('login.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logout berhasil')
    return redirect('/')

# Route index
@app.route('/')
def home():
    return render_template('index.html')

# Route klasifikasi
@app.route('/klasifikasi', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        method = request.form.get('method', 'svm')  # ambil metode dari form, default SVM

        if not file or file.filename == '':
            return jsonify({"error": "No file uploaded"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Only PNG, JPG, and JPEG are allowed."}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Resize dan convert to gray
        img = cv2.imread(filepath)
        img_resized = cv2.resize(img, (128, 128))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        preprocessed_path = os.path.join(UPLOAD_FOLDER, "preprocessed_" + file.filename)
        cv2.imwrite(preprocessed_path, img_gray)

        hog_url=None
        if method == 'svm':
            # Ekstraksi fitur HOG
            hog_url = url_for('static', filename='uploads/' + "hog_" + file.filename)
            hog_features, hog_image = hog(img_gray, pixels_per_cell=(8,8),
                                          cells_per_block=(2,2), visualize=True, feature_vector=True)
            hog_path = os.path.join(UPLOAD_FOLDER, "hog_" + file.filename)
            cv2.imwrite(hog_path, (hog_image*255).astype("uint8"))

            # Reduksi dimensi & prediksi
            features_pca = pca_model.transform(hog_features.reshape(1, -1))
            prediction = svm_model.predict(features_pca)[0]
            class_name = class_map[prediction]
            class_name_lower = class_name.lower()

        elif method == 'cnn':
            # Preprocessing untuk CNN
            cnn_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            pil_img = Image.open(filepath).convert('RGB')
            input_tensor = cnn_transform(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = cnn_model(input_tensor)
                pred_idx = output.argmax(1).item()
                class_name = cnn_classes[pred_idx]  # 'CBB', 'CBSD', dll.
                class_name_lower = class_name.lower()  # 'cbb', 'cbsd', dll.

        # Metrics per metode
        metrics_svm = {
            'cbb': {'accuracy': 88, 'precision': 87, 'recall': 81, 'f1_score': 83},
            'cbsd': {'accuracy': 88, 'precision': 79, 'recall': 82, 'f1_score': 80},
            'cgm': {'accuracy': 88, 'precision': 92, 'recall': 88, 'f1_score': 90},
            'cmd': {'accuracy': 88, 'precision': 90, 'recall': 92, 'f1_score': 91},
            'healthy': {'accuracy': 88, 'precision': 94, 'recall': 74, 'f1_score': 83},
        }

        metrics_cnn = {
            'cbb': {'accuracy': 83, 'precision': 60, 'recall': 50, 'f1_score': 55},
            'cbsd': {'accuracy': 83, 'precision': 89, 'recall': 80, 'f1_score': 84},
            'cgm': {'accuracy': 83, 'precision': 72, 'recall': 78, 'f1_score': 75},
            'cmd': {'accuracy': 83, 'precision': 94, 'recall': 88, 'f1_score': 91},
            'healthy': {'accuracy': 83, 'precision': 62, 'recall': 90, 'f1_score': 74},
        }

        # Setelah klasifikasi
        if method == 'svm':
            class_name = class_map[prediction]
        else:  # cnn
            class_name = cnn_classes[pred_idx]  # 'CBB', 'CBSD', dll.

        class_name_lower = class_name.lower()

        # Ambil metrics sesuai metode
        if method == 'svm':
            metrics = metrics_svm.get(class_name_lower, {'accuracy':0,'precision':0,'recall':0,'f1_score':0})
        else:
            metrics = metrics_cnn.get(class_name_lower, {'accuracy':0,'precision':0,'recall':0,'f1_score':0})

        # Ambil info penyakit
        disease_info = disease_info_map.get(class_name.upper(), "Info tidak tersedia")

        current_user_id = session.get('user_id')
        if current_user_id:
            latitude = request.form.get('latitude')
            longitude = request.form.get('longitude')

            if latitude and longitude:
                location = get_city_from_coords(latitude, longitude)
            else:
                location = "Tidak diketahui"

            history = ClassificationHistory(
                user_id=current_user_id,
                image_path=filepath,
                result=class_name,
                accuracy=metrics.get('accuracy'),
                precision=metrics.get('precision'),
                recall=metrics.get('recall'),
                f1_score=metrics.get('f1_score'),
                location=location,
                method=method    
            )

            db.session.add(history)
            db.session.commit()

        return jsonify({
            "prediction": class_name,
            "info": disease_info,
            "image_url": url_for('static', filename='uploads/' + file.filename),
            "preprocessed_url": url_for('static', filename='uploads/' + "preprocessed_" + file.filename),
            "method": method,
            "hog_url": hog_url,
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1_score": metrics['f1_score'],
        })

    return render_template('klasifikasi.html')

# Route about
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/history')
def history():
    user_id = session.get('user_id')
    username = session.get('username', 'Pengguna')
    if not user_id:
        flash('Silakan login dulu')
        return redirect('/login')
    histories = ClassificationHistory.query.filter_by(user_id=user_id).order_by(asc(ClassificationHistory.timestamp)).all()
    return render_template('history.html', histories=histories, username=username)

if __name__ == '__main__':
    app.run()

