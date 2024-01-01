import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

model = pickle.load(open('salary_class.sav', 'rb'))

st.title('Klasifikasi Gaji Algoritma Decision Tree')


age = st.text_input('Usia Anda')
workclass = st.selectbox('Jenis Pekerjaan anda ?', ['Bekerja Di Perusahaan', 'Freelancer', 'Instansi Pemerintahan Lokal', 'Instansi pemerintahan nasional', 'Magang', 'Instansi Federal (Polisi/Tentara)', 'Tanpa Penghasilan', 'Tidak Bekerja', 'Tidak Tahu'])

if workclass == 'Bekerja Di Perusahaan':
    workclass = 4
elif workclass == 'Freelancer':
    workclass = 6
elif workclass == 'Instansi Pemerintahan Lokal':
    workclass = 2
elif workclass == 'Instansi pemerintahan nasional':
    workclass = 7
elif workclass == 'Magang':
    workclass = 5
elif workclass == 'Instansi Federal (Polisi/Tentara)':
    workclass = 1
elif workclass == 'Tanpa Penghasilan':
    workclass = 8
elif workclass == 'Tidak Bekerja':
    workclass = 3
else:
    workclass = 0

education = st.selectbox('Pendidikan Anda?', ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', '11th', 'Assoc-acdm', '10th', '7th-8th', 'Prof-school', '9th', '12th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool'])
    
if education == 'HS-grad':
    education = 11
elif education == 'Some-college':
    education = 15
elif education == 'Bachelors':
    education = 9
elif education == 'Masters':
    education = 12
elif education == 'Assoc-voc':
    education = 8
elif education == '11th':
    education = 1
elif education == 'Assoc-acdm':
    education = 7
elif education == '10th':
    education = 0
elif education == '7th-8th':
    education = 5
elif education == 'Prof-school':
    education = 14
elif education == '9th':
    education = 6
elif education == '12th':
    education = 2
elif education == 'Doctorate':
    education = 10
elif education == '5th-6th':
    education = 4
elif education == '1st-4th':
    education = 3
else:
    education = 13

marital_status = st.selectbox('Status Pernikahan Anda?', ['Menikah dengan warga sipil', 'Belum Menikah', 'Bercerai', 'Pisah-Ranjang', 'Menjanda/Duda', 'Menikah tetapi pasangan di luar kota', 'Menikah dengan anggota TNI/Polri'])
    
if marital_status == 'Menikah dengan warga sipil':
    marital_status = 2
elif marital_status == 'Belum Menikah':
    marital_status = 4
elif marital_status == 'Bercerai':
    marital_status = 0
elif marital_status == 'Pisah-Ranjang':
    marital_status = 5
elif marital_status == 'Menjanda/Duda':
    marital_status = 6
elif marital_status == 'Menikah tetapi pasangan di luar kota':
    marital_status = 3
else:
    marital_status = 1

occupation = st.selectbox('Jenis Pekerjaan Anda?', ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Tidak Tahu', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'])
    
if occupation == 'Prof-specialty':
    occupation = 10
elif occupation == 'Craft-repair':
    occupation = 3
elif occupation == 'Exec-managerial':
    occupation = 4
elif occupation == 'Adm-clerical':
    occupation = 1
elif occupation == 'Sales':
    occupation = 12
elif occupation == 'Other-service':
    occupatione = 8
elif occupation == 'Machine-op-inspct':
    occupation = 7
elif occupation == 'Tidak Tahu':
    occupation = 0
elif occupation == 'Transport-moving':
    occupation = 14
elif occupation == 'Handlers-cleaners':
    occupation = 6
elif occupation == 'Farming-fishing':
    occupation = 5
elif occupation == 'Tech-support':
    occupation = 13
elif occupation == 'Protective-serv':
    occupation = 11
elif occupation == 'Priv-house-serv':
    occupation = 9
else:
    occupation = 2

relationship = st.selectbox('Status Hubungan Anda?', ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
    
if relationship == 'Husband':
    relationship = 0
elif relationship == 'Not-in-family':
    relationship = 1
elif relationship == 'Own-child':
    relationship = 3
elif relationship == 'Unmarried':
    relationship = 4
elif relationship == 'Wife':
    relationship = 5
else:
    relationship = 2

race = st.selectbox('Pilih Ras Anda?', ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    
if race == 'White':
    race = 4
elif race == 'Black':
    race = 2
elif race == 'Asian-Pac-Islander':
    race = 1
elif race == 'Amer-Indian-Eskimo':
    race = 0
else:
    race = 3
sex = st.selectbox('Jenis Kelamin anda ?', ['Laki Laki', 'Perempuan'])

if sex == 'Laki Laki':
    sex = 1
else:
    sex = 0

capital_gain = st.text_input('Pendapatan Perkapita')
capital_loss = st.text_input('Nilai inflasi Perkapita')
hours_per_week = st.text_input('Jumlah jam kerja perminggu')

region = st.selectbox('Pilih Wilayah Anda?', ['Western', 'Latin American', 'Other', 'Asian'])
    
if region == 'Western':
    region = 3
elif region == 'Latin American':
    region = 1
elif region == 'Other':
    region = 2
else:
    region = 0



kelas = ''

if st.button('Kelasifikasi Gaji'):
    kelas_gaji = model.predict([[age, workclass, education, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, region]])
    
    if(kelas_gaji[0] == 0):
        kelas = 'Gaji Kurang dari sama dengan 50k'
    else :
        kelas ='Gaji Lebih dari 50k'

    st.success(kelas)

if st.button('Visualisasi D3'):
    # Visualize the Decision Tree using matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=['age', 'workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'region'], ax=ax, class_names=['<=50k', '>50k'])
    st.pyplot(fig)