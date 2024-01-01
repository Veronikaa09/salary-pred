# Laporan Proyek Machine Learning
### Nama : Veronika
### Nim : 211351147
### Kelas : Teknik Informatika Pagi B

## Domain Proyek

Pada bagian ini, kamu perlu menuliskan latar belakang yang relevan dengan proyek yang diangkat.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
  
  Format Referensi: [Judul Referensi](https://scholar.google.com/) 

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. dataset wajib menggunakan [kaggle](https://www.kaggle.com/) dan **atribut yang digunakan minimal 8 atribut**. Sertakan juga sumber atau tautan untuk mengunduh dataset.<br> 

Contoh: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- Age : merupakan umur pasien dalam satuan tahun.
- Sex : merupakan jenis kelamin pasien meliputi [M: Male, F: Female].
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pertama kita import dulu library yang di butuh dengan memasukan perintah :
```bash
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree

import pickle
```

Kemudian agar aplikasi bisa berjalan otomatis di collab maka kita harus mengkoneksikan file token kaggle kedalam aplikasi kita dan membuat directory khusus.
```bash
from google.colab import files
files.upload()
```

upload file token kaggle kita kemudian
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

Jika direktori sudah dibuat maka kita bisa mendownload datasetnya
```bash
!kaggle datasets download -d ayessa/salary-prediction-classification
```

Setelah terdownload, kita bisa mengekstrak dataset terserbut dan memindahkan nya kedalam folder yang sudah di buat
```bash
!mkdir salary-prediction-classification
!unzip salary-prediction-classification.zip -d salary-prediction-classification
!ls salary-prediction-classification
```

Jika sudah, maka kita bisa langsung membuka file dataset tersebut
```bash
df = pd.read_csv('/content/salary-prediction-classification/salary.csv')
```

kemudian kita bisa panggil data tersebut, karena saya definisakan dengan df maka saya bisa panggil dengan cara
```bash
df.head()
```

cek tipe data dari masing-masing atribut/fitur dari dataset
```bash
df.info()
```

untuk menampilkan jumlah data dan kolom yang ada di dataset
```bash
df.shape
```
untuk pengelompokan data dan menghitung jumlah kemunculan 
```bash
item_count = df.groupby(["sex", "salary"])["salary"].count().reset_index(name="Count")
item_count.head(10)
```

untuk melihat nilai unik 
``bash
df['native-country'].unique()
```

untuk mengelompokkan negara-negara berdasarkan wilayah atau region tertentu.
```bash
def segment_country(country):
    if country in [' United-States', ' Canada', ' England', ' Germany', ' France', ' Italy', ' Holand-Netherlands', ' Ireland', ' Scotland', ' Portugal', ' Greece']:
        return 'Western'
    elif country in [' India', ' Japan', ' China', ' Hong', ' Taiwan', ' Philippines', ' Vietnam', ' Thailand', ' Laos']:
        return 'Asian'
    elif country in [' Mexico', ' Cuba', ' Jamaica', ' Puerto-Rico', ' Honduras', ' El-Salvador', ' Guatemala', ' Dominican-Republic', ' Nicaragua', ' Peru', ' Columbia']:
        return 'Latin American'
    else:
        return 'Other'

df['region'] = df['native-country'].apply(segment_country)
```
```bash
print("Unique Regions:")
print(df['region'].unique())
```

untuk menghapus kolom native-country
```bash
df.drop(columns = 'native-country', inplace=True)
```

untuk mengubah nilai-nilai kategori pada kolom bertipe data objek menjadi numerik
```bashlabel_encoder = LabelEncoder()

object_columns = df.select_dtypes(include=['object']).columns

for column in object_columns:
    df[column] = label_encoder.fit_transform(df[column])

df.info()
```

## Modeling
Untuk menentukan x (feature) dan y (label) :
```bash
x = df.drop(['salary'],axis=1)
y = df['salary']
```

Untuk memisahkan data training dan data testing dengan memasukan perintah :
```bash
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)
```

Lalu masukan data training dan testing ke dalam model decision tree dan mengecek akurasinya dengan perintah :
```bash
dtc = DecisionTreeClassifier(
    ccp_alpha=0.0, class_weight=None, criterion='entropy',
    max_depth=4, max_features=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_samples_leaf=1,
    min_samples_split=2, min_weight_fraction_leaf=0,
    random_state=42, splitter='best'
)

model = dtc.fit(x_train, y_train)

y_pred = dtc.predict(x_test)

dtc_acc = accuracy_score(y_test, dtc.predict(x_test))

print(f"Tingkat Akurasi data training = {accuracy_score(y_train, dtc.predict(x_train))}")
print(f"Tingkat akurasi data testing = {dtc_acc} \n")
```
```bash
out :
Tingkat Akurasi data training = 0.8444190944190945
Tingkat akurasi data testing = 0.8442010441191524
```
## Evaluation
Metrik evaluasi yang digunakan yaitu confusion matrik dengan memasukan perintah :
```bash
confusion_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=dtc.classes_, yticklabels=dtc.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()
```
![image](https://github.com/Veronikaa09/salary-pred/assets/149310956/18ffae41-1ad1-42c3-bfa8-f6ec71548cfe)

## Visualisasi
untuk mengaplikasikan model machine learning 
```bash
input_data = (39,7,77516,9,13,4,1,1,4,1,2174,0,40,3)

input_data_as_numpy_array = np.array(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print('Gaji Kurang dari sama dengan 50k')
else:
  print('Gaji diatas 50k')
```

untuk visualisasi datanya
```bash
plt.figure(figsize=(40,30))
tree.plot_tree(
    model,
    feature_names=['age','workclass',	'fnlwgt',	'education',	'education-num',	'marital-status	occupation',	'relationship',	'race',	'sex',	'capital-gain	capital-loss',	'hours-per-week	salary','region'],
    class_names = ['<=50k','>50K'],
    filled = True
)
plt.show()
```
![image](https://github.com/Veronikaa09/salary-pred/assets/149310956/8dec68e8-19bd-44d1-a574-d58ab4dafab2)


## Deployment
pada bagian ini anda memberikan link project yang diupload melalui streamlit share. boleh ditambahkan screen shoot halaman webnya.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

