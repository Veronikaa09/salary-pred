# Laporan Proyek Machine Learning
### Nama : Veronika
### Nim : 211351147
### Kelas : Teknik Informatika Pagi B

## Domain Proyek
Data asini didasarkan pada hasil sensus yang dilakukan Barry Baker pada tahun 1994 untuk menentukan jenjang pekerjaan apa saja yang dapat menghasilkan gaji lebih dari $50k pertahun dan pekerjaan apa saja yang mendapat gaji kurang dari $50k

## Business Understanding

berdasarkan hasil penelitian Barry baker yang dilakukan pada tahun 1994 yang didasarkan pada sensus kependudukan, didapat beberapa hasil perihal jenjang pekerjaan dan pendapatan tahunan nya.
Diharapkan dengan menggunakan algoritma Decision Tree kita bisa menenetukan klasifikasi pada pendapatan tahunan, yang di dasarkan pada 14 parameter yang akan di jabarkan di bawah

Bagian laporan ini mencakup:

### Problem Statements

- Ketidaktahuan seseorang terhadap klassifikasi gaji yang akan didapatkan berdasarkan parameter yang telah ditentukan.

### Goals

- Untuk memudahkan kita menentukan jenjang pekerjaan. 

    ### Solution statements
    - Maka dibuatkannya aplikasi Klasifikasi Gaji menggunakan Algoritma Decision Tree.

## Data Understanding
Dataset yang digunakan adalah dataset yang diambil dari kaggle, didalamnya berisi 14 parameter.

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- age: Merupakan umur.
- workclass: Merupakan jenis kelas.
- education: Merupakan jenis pendidikan.
- education-num: Merupakan nomor pendidikan.
- marital-status: Merupakan status pernikahan.
- occupation: Merupakan jenis pekerjaan.
- relationship: Merupakan status hubungan.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Merupakan jenis kelamin.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: Merupakan
  
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
## EDA
untuk menampilkan seberapa sering setiap nilai pada kolom salary muncul dalam dataset.
```bash
sns.countplot(x=df['salary'], data=df)
plt.show()
```
![image](https://github.com/Veronikaa09/salary-pred/assets/149310956/03427b63-daf5-46d7-b327-d4a2efb05a7d)

untuk menunjukan tingkat persentase berdasarkan etnis.
```bash
plt.figure(figsize=(6, 6))
plt.title('Tingkat Persentase berdasarkasn etnis')

count_pekerjaan = df['race'].value_counts()

plt.pie(count_pekerjaan, labels=count_pekerjaan.index, autopct='%1.1f%%', startangle=140)

plt.show()
```
![image](https://github.com/Veronikaa09/salary-pred/assets/149310956/0f46bdc0-623e-4b43-a33c-5dcf01f31a03)

untuk menampilkan jumlah kabupaten per provinsi
```bash
# Menghitung jumlah kabupaten per provinsi
count_jenis_pekerjaan = df['occupation'].value_counts()

# Mengambil top 10 provinsi
top_10_pekerjaan = count_jenis_pekerjaan.head(10)

# Membuat plot menggunakan seaborn dengan rotasi label 45 derajat
plt.figure(figsize=(10, 3))
plot = sns.countplot(x='occupation', data=df, order=top_10_pekerjaan.index)
plot.set_xticklabels(plot.get_xticklabels(), rotation=45, ha='right')  # Menambahkan rotasi 45 derajat
plt.title('Top 10 Pekerjaan')
plt.show()
```
![image](https://github.com/Veronikaa09/salary-pred/assets/149310956/6cd272c6-2f80-4ed5-9907-79c4e4e314c3)

untuk menampilkan penyebaran berdasarkan usia.
```bash
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
plt.title('Penyebaran berdasarkan usia')
plt.xlabel('Usia')
plt.ylabel('Frequency')
plt.show()

plt.show()
```
![image](https://github.com/Veronikaa09/salary-pred/assets/149310956/55d701af-e3a4-40c7-a0b4-8c0b94990f33)

untuk menampilkan jumlah jam kerja perminggu.
```bash
plt.figure(figsize=(10, 6))
sns.histplot(df['hours-per-week'], bins=30, kde=True, color='salmon')
plt.title('Jumlah Jam Kerja Perminggu')
plt.xlabel('Jam Perminggu')
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/Veronikaa09/salary-pred/assets/149310956/b0b195da-8fe7-426a-8ddc-e2890dd35c69)

untuk menampilakan jenjang edukasi berdasarkan jenis kelamin.
```bash
plt.figure(figsize=(12, 8))
sns.countplot(x='education', data=df, hue='sex', palette='muted')
plt.title('Jenjang edukasi bersarkan jenis kelamin')
plt.xlabel('Jenjang edukasi')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
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
```bash
out : [0]
Gaji Kurang dari sama dengan 50k
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
[Klasifikasi Gaji Algoritma Decision Tree](https://salary-pred-d3.streamlit.app/)
![image](https://github.com/Veronikaa09/salary-pred/assets/149310956/1b24aa06-c819-40a1-9a3b-54235c68fe51)

