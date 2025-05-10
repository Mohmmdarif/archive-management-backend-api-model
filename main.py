import pytesseract
import uvicorn
import cv2
import numpy as np
import spacy
import joblib
import imutils
import time
import re
import string
import requests

from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# Model Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



# Path ke model NER
NER_MODEL_FOLDER = "d:/Project/SKRIPSI/api-model/ner_model_sangat_baru"

# Path ke model Klasifikasi
KLASIFIKASI_MODEL_FOLDER = "d:/Project/SKRIPSI/api-model/klasifikasi-tfidf"

# Memuat model NER spaCy
ner_model = spacy.load(NER_MODEL_FOLDER)

# Memuat model TF-IDF dan Random Forest
tfidf_vectorizer = joblib.load(f"{KLASIFIKASI_MODEL_FOLDER}/tfidf_vectorizer.pkl")
random_forest_model = joblib.load(f"{KLASIFIKASI_MODEL_FOLDER}/random_forest_model.pkl")




# Inisialisasi stopword remover & stemmer
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()



# Fungsi Preprocessing Teks untuk Klasifikasi
def preprocess_text_classification(text: str) -> str:
    text = text.lower()  # Case folding
    text = re.sub(r'\d+', ' ', text)  # Hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))  # Hapus tanda baca
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
    text = stopword_remover.remove(text)  # Stopword removal
    text = stemmer.stem(text)  # Stemming
    return text

# Melakukan klasifikasi teks menggunakan TF-IDF dan Random Forest
def classify_text(text: str) -> str:
    # Preprocessing teks untuk klasifikasi
    processed_text = preprocess_text_classification(text)

    # Transformasi teks menggunakan TF-IDF
    text_tfidf = tfidf_vectorizer.transform([processed_text])

    # Prediksi menggunakan model Random Forest
    prediction = random_forest_model.predict(text_tfidf)

    return prediction[0]


# Fungsi untuk ekstraksi entitas menggunakan model NER
def extract_entities(text: str) -> dict:
    doc = ner_model(text)
    print("Entities found:", [(ent.text, ent.label_) for ent in doc.ents])
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities

# Fungsi untuk memotong header dari gambar
def crop_header(gray_img, crop_ratio=0.15):
    h = gray_img.shape[0] 
    cropped = gray_img[int(h * crop_ratio):, :] 
    return cropped

# Fungsi untuk mendeskew gambar
def deskew(image: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

    if lines is None:
        return image

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta)
        if 80 < angle < 100:  # fokus hanya horizontal-ish lines
            angles.append(angle - 90)

    if not angles:
        return image

    median_angle = np.median(angles)
    return imutils.rotate_bound(image, -median_angle)

# Fungsi untuk mengekstrak teks dari PDF menggunakan PyPDF2 `Jika bukan file scan`
def extract_text_pypdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text if text.strip() else None  # Jika teks ditemukan, return teksnya

# Fungsi untuk meningkatkan kualitas gambar sebelum OCR
def preprocess_image(image, header_crop=False):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY) 
    blurred = cv2.GaussianBlur(gray, (3,3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)

    deskewed = deskew(enhanced)
    
    processed = crop_header(deskewed) if header_crop else deskewed

    _, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binarisasi

    # Konversi kembali ke format gambar PIL untuk OCR
    final_img = Image.fromarray(thresh)

    return final_img


# Fungsi untuk mengekstrak teks dari PDF (OCR jika scan, langsung jika teks ada)
def extract_text_from_pdf(pdf_path):
    custom_config = r'--oem 3 --psm 4 -l ind'

    start_text_extraction = time.time()
    # Coba ekstraksi teks langsung
    extracted_text = extract_text_pypdf(pdf_path)
    end_text_extraction = time.time()
    if extracted_text:
        print(f"Waktu ekstraksi teks: {end_text_extraction - start_text_extraction: .2f} detik")
        return extracted_text

    # Jika tidak ada teks, lakukan OCR
    images = convert_from_path(pdf_path, dpi=400)
    text_results = []

    start_ocr = time.time()
    # Lakukan OCR pada setiap halaman
    for i, img in enumerate(images):
        print(f"Processing page {i+1}...")

        if i == 0:
            preprocessed_img = preprocess_image(img, header_crop=True)
        else:
            preprocessed_img = preprocess_image(img)

        text = pytesseract.image_to_string(preprocessed_img, lang='ind', config=custom_config)
        text_results.append(text)

    end_ocr = time.time()
    print(f"Waktu OCR: {end_ocr - start_ocr: .2f} detik\n")

    return "\n".join(text_results)


# Function Utils

# Membagi hasil klasifikasi teks menjadi beberapa bagian
def slice_text_classification_result(text):
    # Split berdasarkan tanda hubung (-)
    parts = text.split("- surat")
    # Hapus spasi tambahan di setiap bagian
    parts = [part.strip() for part in parts]
    return parts


app = FastAPI()

class FileRequest(BaseModel):
    file_url: str

@app.post("/file")
async def upload_file(request: FileRequest):
    file_url = request.file_url
    print("URL file:", file_url)
    try:
        # Unduh file dari URL Cloudinary
        response = requests.get(file_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download file from URL")

        # Simpan file sementara di disk
        temp_file_path = "temp_file.pdf"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(response.content)

        # OCR
        ocrResult = extract_text_from_pdf(temp_file_path)

        # Ekstraksi entitas (NER)
        entities = extract_entities(ocrResult)

        # Klasifikasi
        classification_result = classify_text(ocrResult)

        classification = slice_text_classification_result(classification_result)

        # Hapus file sementara
        Path(temp_file_path).unlink()

        return {
            "classification": [
                {
                    "Classify": classification[0],
                    "Criteria": classification[1]
                }
            ],
            "entities": entities,
            "text": ocrResult
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

