import streamlit as st
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import re
import string
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Setup NLTK dan Sastrawi
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Judul aplikasi
st.markdown("<h1 style='text-align: center;'>PREDIKSI SENTIMEN BERITA KEUANGAN MEDIA DI INDONESIA</h1>", unsafe_allow_html=True)

# Load pre-trained BERT model dan tokenizer
tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')
MAX_LEN = 128

# Definisikan ulang model sesuai arsitektur yang digunakan sebelumnya
class SentimentBERTModel(torch.nn.Module):
    def __init__(self, dropout=0.3):
        super(SentimentBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 4)  # 4 classes: Negative, Neutral, Positive, Dual

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        fc1_output = self.fc1(dropout_output)
        relu_output = self.relu(fc1_output)
        output = self.fc2(relu_output)
        return output

# Inisialisasi model
model = SentimentBERTModel()
model.load_state_dict(torch.load('bert_model_state_dict.pt'))
model.eval()

# Preprocessing function
def clean_text(text):
    # Menghapus angka, simbol, URL, dan tanda baca
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\.com|\.id|\.co', '',regex=True)
    text = re.sub(r'https?\S+|www\.\S+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[{}]+'.format(string.punctuation), '', text)

    # Menghapus tambahan simbol
    additional_symbols = r'[©â€“œ]'
    text = re.sub(additional_symbols, '', text)

    # Menghapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text)

    # Tokenisasi kata
    words = nltk.word_tokenize(text)

    # Menghapus stopwords bahasa Indonesia
    words = [word.lower() for word in words if word.lower() not in stop_words]

    # Stemming menggunakan Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

# Fungsi prediksi
def predict(text, model, tokenizer):
    cleaned_text = clean_text(text)  # Preprocess teks inputan
    inputs = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs, dim=1)
    return prediction.item()

# Mengurangi jarak secara maksimal antara label dan text box
st.markdown('<p style="font-size: 20px; margin-top: 15px; margin-bottom: -30px; line-height: 1;">Masukkan Teks Berita :</p>', unsafe_allow_html=True)

input_text = st.text_area("", height=300)

# CSS untuk merapikan tombol
st.markdown("""
    <style>
        .stButton>button {
            background-color: #74574F;
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            margin-left: 545px;
            margin-top: -10px;
            padding: 10px 18px;
            font-size: 30px;
            cursor: pointer;
        }
        .result {
            font-family: 'Times New Roman';
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Jika tombol ditekan, lakukan prediksi
if st.button("Prediksi Sentimen"):
    if input_text:
        hasil_sentimen = predict(input_text, model, tokenizer)
        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive', 3: 'Dual'}

        # Menentukan warna berdasarkan hasil prediksi
        if hasil_sentimen == 0:  # Negative
            color = "#9b373d"  # Warna merah
        elif hasil_sentimen == 1:  # Neutral
            color = "#004278"  # Warna biru
        elif hasil_sentimen == 2:  # Positive
            color = "#006c4f"  # Warna hijau
        elif hasil_sentimen == 3:  # Dual
            color = "#4d4e56"  # Warna emas (gold)

        # Menampilkan hasil dengan warna yang berbeda
        st.markdown(f"""
        <div style="border-radius: 0px; padding : 10px; background-color: {color}; color: white;" class="result">
            Hasil Prediksi : {label_map.get(hasil_sentimen, 'Unknown')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.write("Silakan masukkan teks berita untuk diprediksi.")
