# Employee Attrition Prediction

## Project Overview
Proyek ini bertujuan untuk memprediksi kemungkinan karyawan mengundurkan diri (attrition) dan memahami faktor-faktor utama yang mempengaruhinya. Tingginya angka pengunduran diri dapat menimbulkan kerugian finansial dan hilangnya talenta penting dalam perusahaan.

Proyek ini bertujuan untuk menganalisis faktor-faktor yang memengaruhi keputusan karyawan untuk mengundurkan diri (attrition) dan membangun model prediktif berbasis machine learning. Analisis ini diharapkan dapat membantu tim HR dalam mengambil kebijakan retensi yang lebih tepat.
## Latar Belakang

Turnover karyawan merupakan tantangan besar bagi perusahaan. Biaya yang ditimbulkan mencakup pelatihan, perekrutan ulang, hingga penurunan produktivitas. Oleh karena itu, diperlukan analisis untuk memahami faktor-faktor pemicu agar strategi pencegahan dapat dilakukan.

## Business Understanding  

Perusahaan ingin menurunkan angka pengunduran diri dengan memahami siapa saja karyawan yang berpotensi resign dan faktor-faktor apa yang mempengaruhinya. Hal ini penting untuk menyusun kebijakan SDM seperti promosi, kompensasi, dan perbaikan lingkungan kerja.

Attrition karyawan yang tinggi merupakan salah satu tantangan besar dalam manajemen SDM. Perusahaan perlu memahami karakteristik karyawan yang berisiko tinggi mengundurkan diri agar dapat mengambil tindakan preventif seperti meningkatkan kepuasan kerja atau memberikan promosi tepat waktu.

### Problem Statements  
Perusahaan tidak mengetahui secara pasti siapa saja karyawan yang berpotensi mengundurkan diri dan faktor apa yang mendorong mereka keluar. Kurangnya informasi ini menyulitkan perencanaan strategi retensi yang efektif.

### Goals  
Memprediksi kemungkinan seorang karyawan akan resign menggunakan machine learning.

Mengidentifikasi fitur-fitur utama yang berkontribusi terhadap attrition.

Memberikan insight actionable bagi tim HR untuk meningkatkan retensi. 

### Solution Statements  

Melakukan EDA untuk menemukan insight dari data historis.

Membersihkan dan mempersiapkan data (preprocessing & feature engineering).

Melatih model prediktif (Logistic Regression dan Random Forest).

Mengevaluasi performa model dan menginterpretasi fitur penting.

## Data Understanding
# Data Understanding

Dataset yang digunakan dalam proyek ini merupakan data karyawan dari sebuah perusahaan fiktif yang dibuat oleh tim data scientist IBM dan tersedia secara publik di [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
. Dataset ini mencakup informasi demografis, status pekerjaan, riwayat kerja, kepuasan kerja, serta status pengunduran diri (attrition) dari 1.470 karyawan.

Atribut-atribut yang tersedia dijelaskan pada Tabel 1 berikut.

## Tabel 1. Atribut Dataset

| No | Atribut                  | Keterangan                                      | Tipe Data |
| -- | ------------------------ | ----------------------------------------------- | --------- |
| 0  | Age                      | Usia karyawan                                   | int64     |
| 1  | Attrition                | Target apakah karyawan resign (Yes/No)          | object    |
| 2  | BusinessTravel           | Frekuensi perjalanan bisnis                     | object    |
| 3  | DailyRate                | Gaji harian                                     | int64     |
| 4  | Department               | Departemen tempat bekerja                       | object    |
| 5  | DistanceFromHome         | Jarak dari rumah ke kantor (dalam satuan mil)   | int64     |
| 6  | Education                | Tingkat pendidikan (1–5)                        | int64     |
| 7  | EducationField           | Bidang pendidikan                               | object    |
| 8  | EmployeeCount            | Jumlah karyawan (selalu 1 → fitur konstan)      | int64     |
| 9  | EmployeeNumber           | ID unik karyawan                                | int64     |
| 10 | EnvironmentSatisfaction  | Kepuasan terhadap lingkungan kerja (1–4)        | int64     |
| 11 | Gender                   | Jenis kelamin                                   | object    |
| 12 | HourlyRate               | Gaji per jam                                    | int64     |
| 13 | JobInvolvement           | Tingkat keterlibatan pekerjaan (1–4)            | int64     |
| 14 | JobLevel                 | Level jabatan                                   | int64     |
| 15 | JobRole                  | Nama jabatan                                    | object    |
| 16 | JobSatisfaction          | Kepuasan terhadap pekerjaan (1–4)               | int64     |
| 17 | MaritalStatus            | Status pernikahan                               | object    |
| 18 | MonthlyIncome            | Gaji bulanan                                    | int64     |
| 19 | MonthlyRate              | Gaji bulanan (rate-based)                       | int64     |
| 20 | NumCompaniesWorked       | Jumlah perusahaan sebelumnya                    | int64     |
| 21 | Over18                   | Apakah berusia >18 (selalu ‘Y’ → fitur konstan) | object    |
| 22 | OverTime                 | Apakah bekerja lembur                           | object    |
| 23 | PercentSalaryHike        | Persentase kenaikan gaji                        | int64     |
| 24 | PerformanceRating        | Rating performa kerja                           | int64     |
| 25 | RelationshipSatisfaction | Kepuasan relasi interpersonal di tempat kerja   | int64     |
| 26 | StandardHours            | Jam kerja standar (selalu 80 → fitur konstan)   | int64     |
| 27 | StockOptionLevel         | Level opsi saham                                | int64     |
| 28 | TotalWorkingYears        | Total tahun pengalaman kerja                    | int64     |
| 29 | TrainingTimesLastYear    | Frekuensi pelatihan dalam setahun terakhir      | int64     |
| 30 | WorkLifeBalance          | Keseimbangan kerja-hidup (1–4)                  | int64     |
| 31 | YearsAtCompany           | Lama bekerja di perusahaan saat ini             | int64     |
| 32 | YearsInCurrentRole       | Lama menjabat posisi saat ini                   | int64     |
| 33 | YearsSinceLastPromotion  | Lama sejak promosi terakhir                     | int64     |
| 34 | YearsWithCurrManager     | Lama bekerja dengan atasan saat ini             | int64     |

### Exploratory Data Analysis (EDA)
1. Berapa Persentase Karyawan yang Resign?

    Dari keseluruhan data, sekitar 16% karyawan menyatakan resign (label Attrition = Yes). Meskipun mayoritas karyawan tetap bertahan, angka ini tidak bisa diabaikan karena tingkat attrition yang tinggi dapat berdampak signifikan terhadap biaya operasional, produktivitas, dan stabilitas tim di perusahaan.

    ![Persentase Karyawan Resign](image/1.png)
  

2. Apakah Usia Muda Lebih Sering Resign?

    Karyawan berusia di bawah 31 tahun menunjukkan tingkat pengunduran diri tertinggi, yaitu mencapai 38%. Fakta ini menandakan bahwa kelompok usia muda cenderung lebih aktif dalam mencari peluang karier baru atau merasa kurang cocok dengan lingkungan kerja saat ini, sehingga keputusan untuk resign terjadi lebih awal dalam masa kerja mereka.

   ![Distribusi Usia Berdasarkan Status Attrition](image/PKR.png)
     Gambar 2. Distribusi Usia Berdasarkan Status Attrition
  
3. Apa Pengaruh Marital Status terhadap Resign?

    Status pernikahan turut memengaruhi kecenderungan karyawan untuk resign. Karyawan yang berstatus single mencatat tingkat attrition tertinggi sebesar 25.5%, jauh lebih tinggi dibandingkan dengan mereka yang sudah menikah (12.5%) atau bercerai (10.1%). Hal ini bisa jadi karena fleksibilitas dalam mobilitas karier atau minimnya tanggungan keluarga, sehingga mempermudah keputusan untuk berpindah pekerjaan.

   ![Attrition Berdasarkan Status Pernikahan](image/SP.png)
   Gambar 3. Attrition Berdasarkan Status Pernikahan

4. Apakah Jarak ke Kantor Mempengaruhi Keputusan Resign?

    Jarak tempuh antara rumah dan kantor ternyata berperan dalam keputusan resign. Karyawan yang tinggal lebih jauh menunjukkan kecenderungan attrition yang lebih tinggi. Faktor seperti lamanya perjalanan, biaya transportasi, dan dampaknya terhadap keseimbangan hidup dan kerja, kemungkinan besar memicu ketidakpuasan yang berujung pada keputusan untuk meninggalkan perusahaan

    ![Jarak dari Rumah Berdasarkan Status Attrition](image/3.png)
  Gambar 4. Jarak dari Rumah Berdasarkan Status Attrition
   
5. Job Role Mana yang Paling Banyak Resign?

    Peran pekerjaan dengan tingkat attrition tertinggi adalah Sales Representative. Hal ini dapat dikaitkan dengan tekanan target yang tinggi, dinamika pasar yang fluktuatif, atau tuntutan kerja yang lebih kompetitif. Profesi ini sering kali memiliki turnover rate yang tinggi karena karakteristik pekerjaan yang menantang dan intens.

    ![Tingkat Attrition Berdasarkan Job Role](image/4.png)
    Gambar 5. Tingkat Attrition Berdasarkan Job Role

6. Apakah Level Jabatan (JobLevel) Mempengaruhi Attrition?
    
    Tingkat jabatan (JobLevel) menunjukkan hubungan langsung dengan tingkat attrition. Karyawan pada level entry-level memiliki kecenderungan resign yang lebih tinggi dibandingkan dengan mereka yang berada di jenjang karier menengah atau atas. Hal ini bisa dipicu oleh keinginan untuk mencari kenaikan jabatan, ekspektasi awal yang tidak terpenuhi, atau keterbatasan ruang tumbuh dalam posisi mereka saat ini.
    
    ![Attrition Berdasarkan Job Level](image/5.png)
    Gambar 6. Attrition Berdasarkan Job Level

7. Apakah Karyawan Baru (<2 Tahun) Lebih Rentan Resign?

    Karyawan dengan masa kerja kurang dari dua tahun memiliki tingkat resign yang jauh lebih tinggi dibandingkan mereka yang sudah bekerja lebih lama. Hal ini menandakan bahwa fase awal pekerjaan merupakan masa kritis dalam retensi karyawan, di mana faktor adaptasi, budaya perusahaan, dan kepuasan awal sangat memengaruhi keputusan untuk bertahan atau tidak.

    ![Attrition Berdasarkan Lama Bekerja di Perusahaan](image/6.png)
  Gambar 7. Attrition Berdasarkan Lama Bekerja di Perusahaan

8. Apakah Gaji Rendah Berkorelasi dengan Attrition?
    
    Hasil analisis menunjukkan bahwa karyawan dengan gaji bulanan yang lebih rendah memiliki probabilitas resign yang lebih tinggi. Keterbatasan penghasilan dapat memicu ketidakpuasan, terutama jika tidak sebanding dengan beban kerja atau kebutuhan hidup. Sebaliknya, gaji yang kompetitif cenderung memberikan rasa aman dan loyalitas terhadap perusahaan.

    ![Distribusi Monthly Income Berdasarkan Status Attrition](image/7.png)
  Gambar 8. Distribusi Monthly Income Berdasarkan Status Attrition

9. Apakah Ada Perbedaan Attrition antar Department?
    
    Setiap departemen menunjukkan tingkat attrition yang berbeda-beda. Departemen Sales memiliki persentase angka pengunduran diri yang lebih tinggi, kemungkinan karena tingginya tekanan kinerja dan tuntutan pencapaian target. Sementara itu, departemen dengan stabilitas kerja yang lebih tinggi cenderung mempertahankan karyawan lebih lama, berkat struktur kerja yang mendukung dan jalur karier yang lebih jelas.

    ![Tingkat Attrition Berdasarkan Department](image/8.png)
  Gambar 9. Tingkat Attrition Berdasarkan Department

10. Apakah Pengalaman Pendek (TotalWorkingYears) Mempengaruhi Resign?
    
    Tingkat attrition lebih tinggi ditemukan pada karyawan dengan pengalaman kerja (Total Working Years) yang lebih pendek. Hal ini wajar, karena karyawan dengan pengalaman terbatas sering kali masih dalam proses eksplorasi karier dan belum menemukan kecocokan ideal. Sebaliknya, pengalaman yang lebih panjang umumnya mencerminkan kematangan profesional dan komitmen yang lebih kuat terhadap perusahaan.

    ![Attrition Berdasarkan Total Working Years](image/9.png)
  Gambar 10. Attrition Berdasarkan Total Working Years

11. Apakah Kurangnya Promosi Mempengaruhi Keputusan Resign?

    Kurangnya promosi menjadi salah satu faktor penting dalam keputusan resign. Karyawan yang tidak mengalami peningkatan posisi dalam jangka waktu lama cenderung merasa stagnan dan tidak dihargai, sehingga memilih mencari peluang pertumbuhan di tempat lain. Sebaliknya, kesempatan naik jabatan mampu meningkatkan motivasi dan retensi karyawan secara signifikan.

    ![Attrition Berdasarkan Lama Tidak Dipromosikan](image/10.png)
  Gambar 11. Attrition Berdasarkan Lama Tidak Dipromosikan

## Data Preparation

![Film Terpopuler](image/pp.png)

## Modeling & Result

Berikut adalah hasil evaluasi model yang digunakan untuk prediksi customer churn:

| **Model**                   | **Akurasi**        | **Recall**        | **AUC Train**       | **AUC Test**       |
|-----------------------------|:-----------------:|:-----------------:|:------------------:|:-----------------:|
| **Logistic Regression**     | 🟡 88%            | 🔴 60%            | 🟠 82%             | 🟠 84%            |
| **Random Forest**           | **🟢 95%**        | **🟢 89%**        | **🟢 94%**         | **🟢 92%**        |
| **K-Nearest Neighbors**     | 🟡 90%            | 🟠 68%            | **🟢 96%**         | 🔴 83%            |
| **XGBOOST**                 | 🔴 79%            | 🟡 81%            | 🟡 89%             | 🟡 86%            |
| **SVC**                     | 🟠 84%            | 🟠 85%            | 🟠 91%             | 🟠 91%            |


Recall berfokus pada kemampuan model dalam mengidentifikasi pelanggan yang benar-benar churn (**True Positive, TP**) dari keseluruhan pelanggan yang seharusnya terdeteksi sebagai churn (**TP + False Negative, FN**). Model dengan **recall tinggi** mampu mengenali sebagian besar pelanggan yang berpotensi berhenti berlangganan, memungkinkan perusahaan untuk **mengambil tindakan pencegahan secara proaktif**, seperti menawarkan promosi khusus, meningkatkan layanan, atau memberikan penawaran yang lebih menarik guna mempertahankan pelanggan.

Selain itu, model dengan **nilai FN yang rendah** menunjukkan **ketepatan tinggi dalam menghindari kesalahan pengelompokan pelanggan non-churn sebagai churn**. Dengan demikian, perusahaan tidak akan membuang sumber daya pada pelanggan yang sebenarnya tidak berniat untuk berhenti berlangganan. Kesalahan prediksi yang lebih banyak pada **False Positive (FP)**—di mana pelanggan yang sebenarnya tetap berlangganan diprediksi churn—dapat lebih ditoleransi dibandingkan kesalahan pada FN, karena tindakan mitigasi tetap dapat memberikan manfaat dalam meningkatkan loyalitas pelanggan.  

![Film Terpopuler](image/cm.png)

Secara keseluruhan, model dengan **recall tinggi** menjadi **pilihan terbaik** dalam analisis customer churn, terutama ketika tujuan utama perusahaan adalah **meminimalkan kehilangan pelanggan dan meningkatkan strategi retensi**.

---
Berdasarkan hasil evaluasi model, algoritma **Random Forest Classification** memiliki performa terbaik dengan **akurasi 95%**, **recall 89%**, serta **AUC Train 94%** dan **AUC Test 92%**. Dengan performa yang unggul dalam mengidentifikasi pelanggan yang benar-benar churn, model ini dipilih sebagai model utama untuk pengujian data.  

Berdasarkan hasil pengujian pada **data test**, model **Random Forest Classification** memprediksi bahwa sebanyak **539 pelanggan akan melakukan churn**, sedangkan **211 pelanggan diprediksi tetap berlangganan**. Jumlah pelanggan yang diperkirakan churn ini menunjukkan **persentase yang signifikan** dalam keseluruhan dataset, yang mengindikasikan pentingnya strategi mitigasi untuk mempertahankan pelanggan.  

Dengan informasi ini, perusahaan dapat **menggunakan pendekatan yang lebih personal** untuk mengurangi angka churn, seperti menawarkan **promosi eksklusif**, **meningkatkan kualitas layanan**, atau **mengidentifikasi faktor utama yang menyebabkan ketidakpuasan pelanggan**. Evaluasi lebih lanjut juga diperlukan untuk memastikan **akurasi prediksi** dan **mengurangi kemungkinan kesalahan klasifikasi**, sehingga strategi yang diterapkan dapat lebih efektif dan tepat sasaran.

## Strategi Retensi Pelanggan  

### 1. Segmentasi Pelanggan Berdasarkan Lokasi dan Tingkat Churn  
- **Prioritaskan strategi retensi pelanggan** di lima negara bagian dengan tingkat churn minimal **5%**, yaitu **WV, MN, ID, AL, dan VA**.  
- **Lakukan analisis mendalam** untuk memahami penyebab churn di wilayah tersebut dan menyesuaikan strategi retensi secara lokal.  
- **Bangun program loyalitas** atau penawaran khusus untuk pelanggan di area-area tersebut guna meningkatkan retensi.  

### 2. Optimalisasi Penawaran dan Layanan Berdasarkan Kebiasaan Penggunaan  
- **Evaluasi kembali biaya roaming dan kualitas jaringan** bagi pelanggan dengan **International Plan** dan lakukan perbaikan yang diperlukan.  
- **Tawarkan paket roaming internasional** yang lebih terjangkau atau tingkatkan kualitas jaringan di wilayah-wilayah tertentu.  
- **Berikan insentif atau paket khusus** bagi pelanggan yang sering melakukan panggilan internasional, karena mereka cenderung berbicara lebih lama.  

### 3. Peningkatan Kualitas Layanan dan Pengalaman Pelanggan  
- **Promosikan penggunaan voice mail plan** dengan insentif khusus, meningkatkan kesadaran pelanggan akan manfaatnya.  
- **Identifikasi dan perbaiki masalah utama** yang sering dilaporkan oleh pelanggan yang menghubungi **customer service lebih dari 4 kali**, guna meningkatkan kepuasan pelanggan.  

### 4. Penyesuaian Strategi Harga Berdasarkan Kebutuhan Pelanggan  
- **Tinjau kembali struktur harga panggilan pagi hari**, terutama untuk panggilan dengan durasi panjang, guna mengurangi churn.  
- **Perkenalkan paket atau diskon khusus** untuk panggilan pagi hari agar lebih sesuai dengan kebutuhan pelanggan dan memberikan nilai tambah yang jelas.  
- **Sediakan opsi harga yang kompetitif** untuk panggilan internasional, sambil tetap mempertahankan profitabilitas perusahaan.  

