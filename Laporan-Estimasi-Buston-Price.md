 # Laporan Proyek Machine Learning
### Nama : Aulia Marshanda
### Nim : 211351034
### Kelas : Pagi B

## Domain Proyek

Estimasi Boston House Prices (Harga Rumah Boston) ini dapat digunakan sebagai patokan untuk orang yang ingin membeli rumah di Boston

## Business Understanding

Bisa menghemat biaya dan waktu agar orang tidak pergi ke boston terlebih dahulu  

### Problem Statements

Ketidakmungkinan seseorang untuk pergi langsung ke boston hanya untuk mencari tahu faktor apa saja yang dapat mempengaruhi harga rumah di boston

### Goals

- Mengembangkan model prediksi untuk memudahkan orang yang mencari patokan harga rumah berdasarkan atribut-atribut yang tersedia dalam dataset ini.
- Memahami hubungan dan pengaruh dari berbagai faktor seperti tingkat kejahatan, kualitas lingkungan, dan aksesibilitas terhadap harga rumah di berbagai wilayah di Boston.

### Solution statements

-  Mengembangkan model prediksi penerimaan yang akurat berdasarkan atribut kandidat, yang mengintegrasikan data dari Kaggle.com untuk memberikan saran dan informasi terkait estimasi harga rumah di boston kepada pengguna

- Model yang dihasilkan dari dataset itu menggunakan metode Linear Regression

## Data Understanding

Dataset yang saya gunakan berasal dari kaggle yang berisi faktor-faktor yang mempengaruhi harga rumah di Boston

Kaggle : [Boston House Prices] (https://www.kaggle.com/datasets/vikrishnan/boston-house-prices)


### Variabel-variabel pada Telco Customer Churn Dataset adalah sebagai berikut:

- ZN      : menunjukan Persentase lahan perumahan yang di zonasi untuk lahan hunian (tanpa lahan komersial atau industri). [Tipe Data : Float]
- INDUS   : menunjukan Persentase lahan yang di zonasi untuk penggunaan industri..[Tipe Data : Float]
- CHAS    : menunjukan  apakah properti berbatasan dengan Sungai Charles atau tidak[Tipe Data : Integer]
- NOX     : menunjukan Konsentrasi nitrogen oksida (NOX) di udara.[Tipe Data : Float]
- RM      : menunjukan Rata-rata jumlah kamar per hunian.[Tipe Data : Float]
- AGE     : menunjukan Persentase hunian yang dimiliki sebelum tahun 1940.[Tipe Data : Float]
- DIS     : menunjukan Jarak terbobot dari lima pusat kerja di Boston.
RAD: Indeks aksesibilitas jalan raya.[Tipe Data : Float]
- RAD     : manunjukan Indeks aksesibilitas jalan raya.[Tipe Data : Integer]
- TAX     : menunjukan Tarif pajak properti.[Tipe Data : Float]
- PTRATIO : menunjukan Rasio murid-guru di distrik sekolah.[Tipe Data : Float]      
- B - 1000: menunjukan persentase orang kulit hitam di wilayah tersebut.[Tipe Data : Float]   
- LSTAT   : menunjukan Persentase penduduk dengan status ekonomi rendah.[Tipe Data : Float]      
- MEDV    : Nilai median rumah yang ditempati pemilik dalam $1000.
Kita dapat melihat bahwa atribut input memiliki campuran unit.

## Data Preparation

## Data Collection
Untuk data collection ini, saya mendapatkan dataset yang nantinya digunakan dari website kaggle dengan nama Admission Prediction & jika Anda tertarik dengan datasetnya, Anda bisa click link diatas.

## Data Discovery And Profilling

Untuk bagian ini, kita akan menggunakan teknik EDA.
Pertama kita mengimport semua library yang dibutuhkan,

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

Karena kita menggunakan vscode untuk mengerjakan maka file lokal harus berada di direktori yang sama,

df = pd.read_csv('boston.csv')
df.head()

Lalu tipe data dari masing-masing kolom, kita bisa menggunakan properti info,

df.info()

Selanjutnya kita akan memeriksa apakah dataset tersebut terdapat baris yang kosong atau null dengan menggunakan seaborn,

sns.heatmap(df.isnull())

Selanjutnya mari kita lanjutkan dengan data exploration,

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)

## Modeling

Model regresi linier adalah pendekatan statistik yang digunakan untuk memodelkan hubungan linier

- Sebelumnya mari kita import library yang nanti akan digunakan,

from sklearn.model_selection 
from sklearn.linear_model

- Langkah pertama adalah memasukan kolom-kolom fitur yang ada di datasets dan juga kolom targetnya,

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B - 1000', 'LSTAT', 'MEDV']
df = pd.read_csv('boston.csv', header=None, delimiter=r"\s+", names=column_names)
df.head()
        

- Pembagian X dan Y menjadi train dan testnya masing-masing,

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

- Mari kita lanjut dengan membuat model,

model = LinearRegression()

- Mari lanjut, memasukkan x_train dan y_train pada model dan memasukan X_train_pred,

model.fit(X_train, y_train)

y_pred = model.predict(X_train)

- Sekarang kita bisa melihat score dari model kita,

score = model.score(X_train, y_train)
score

- Akurasi modelnya yaitu 74%, selanjutnya mari kita test menggunakan sebuah array value, 

input_data = (2.31, 0, 0.430483, 6.575, 4.192680, 5.693732, 1.788421)
input_data_np = np.array(input_data)
input_data_reshape = input_data_np.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)

prediksi = model.predict(input_data_reshape)
print(prediksi)

- Sekarang modelnya sudah selesai, mari kita export sebagai file sav agar nanti bisa kita gunakan pada project web streamlit kita,

import pickle

filename = 'PrediksiBoston3.sav'
pickle.dump(model, open(filename, 'wb'))

## Evaluation

Metrik yang digunakan yaitu metrik evaluasi, menggunakan R2 Score.
R2 Score atau juga dikenal sebagai koefisien determinasi, adalah metrik evaluasi umum yang digunakan dalam regresi untuk mengukur sejauh mana model berhasil menjelaskan variasi dalam data. Skor ini memberikan informasi tentang persentase variasi

#Library evaluasi
from sklearn.metrics import r2_score
r2_score(y_train, y_pred)

Hasil yang didapatkan adalah 74%

## Deployment

https://github.com/auliamarshanda12/AuliaMarshanda12
https://auliamarshanda12-isnltufusczhtytsvdxx33.streamlit.app/






