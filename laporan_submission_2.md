# Laporan Proyek Machine Learning - Silmi Azdkiatul Athqia

## Project Overview

Light novel merupakan salah satu bentuk literatur populer yang berkembang pesat, khususnya genre isekai yang menceritakan protagonis yang berpindah ke dunia lain. Dengan semakin banyaknya judul light novel yang tersedia, pembaca sering mengalami kesulitan dalam menemukan novel yang sesuai dengan preferensi mereka. Fenomena information overload ini menciptakan kebutuhan akan sistem rekomendasi yang dapat membantu pembaca menemukan light novel yang relevan berdasarkan minat dan preferensi bacaan mereka.

Sistem rekomendasi telah terbukti efektif dalam berbagai domain seperti e-commerce, streaming musik, dan film. Dalam konteks light novel, sistem rekomendasi dapat membantu meningkatkan pengalaman pembaca dengan menyediakan saran bacaan yang personal dan relevan. Hal ini tidak hanya bermanfaat bagi pembaca, tetapi juga bagi publisher dan platform distribusi digital untuk meningkatkan engagement dan kepuasan pengguna.

**Referensi:**

- Ricci, F., Rokach, L., & Shapira, B. (2015). _Recommender Systems Handbook_. Springer Science & Business Media.
- Isinkaye, F. O., Folajimi, Y. O., & Ojokoh, B. A. (2015). Recommendation systems: Principles, methods and evaluation. _Egyptian Informatics Journal_, 16(3), 261-273.

## Business Understanding

Industri light novel, khususnya genre isekai, mengalami pertumbuhan yang signifikan dengan ribuan judul yang tersedia di berbagai platform. Namun, pembaca sering menghadapi tantangan untuk menemukan novel yang sesuai dengan preferensi mereka dari katalog yang sangat besar.

### Problem Statements

Berdasarkan analisis kebutuhan industri dan pengguna, terdapat beberapa permasalahan utama:

1. **Information Overload**: Pembaca kesulitan menemukan light novel yang relevan dari ribuan judul yang tersedia, menyebabkan frustrasi dan potensi kehilangan minat membaca.

2. **Kurangnya Personalisasi**: Platform existing umumnya hanya menyediakan kategori umum tanpa rekomendasi yang dipersonalisasi berdasarkan preferensi individual pembaca.

3. **Discovery Challenge**: Pembaca cenderung terjebak dalam "filter bubble" dan kesulitan menemukan novel baru yang mungkin sesuai dengan selera mereka namun belum pernah mereka ketahui.

### Goals

Untuk mengatasi permasalahan di atas, proyek ini bertujuan untuk:

1. **Mengembangkan sistem rekomendasi otomatis** yang dapat memberikan saran light novel yang relevan berdasarkan preferensi pembaca, mengurangi waktu pencarian dan meningkatkan kepuasan pengguna.

2. **Menciptakan pengalaman personalisasi** melalui algoritma content-based filtering yang menganalisis karakteristik novel (genre, deskripsi, tema) untuk memberikan rekomendasi yang sesuai dengan profil pembaca.

3. **Meningkatkan discoverability** dengan menyediakan rekomendasi yang seimbang antara relevansi tinggi dan novelty, membantu pembaca menemukan light novel baru yang mungkin mereka sukai.

### Solution Statements

Untuk mencapai goals yang telah ditetapkan, proyek ini akan mengimplementasikan dua pendekatan sistem rekomendasi:

1. **Content-Based Filtering menggunakan TF-IDF dan Cosine Similarity**: Pendekatan ini akan menganalisis konten tekstual dari deskripsi, genre, dan judul novel untuk menemukan kesamaan antar novel. Algoritma TF-IDF akan digunakan untuk mengekstrak fitur teks, kemudian cosine similarity akan menghitung tingkat kesamaan antar novel.

2. **Hybrid Approach dengan Genre-based Weighting**: Sebagai enhancement, sistem akan mengintegrasikan informasi genre dengan pembotan khusus untuk meningkatkan akurasi rekomendasi, mengkombinasikan similarity berbasis konten dengan kategori eksplisit.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **Isekai Light Novel Titles and Descriptions** yang diperoleh dari Kaggle dengan URL: https://www.kaggle.com/datasets/andy8744/isekai-light-novel-titles-and-descriptions

Dataset ini berisi informasi komprehensif tentang 1.366 light novel genre isekai dengan 4 fitur utama. Data ini dipilih karena kelengkapan informasi dan relevansinya dengan genre yang populer di kalangan pembaca light novel.

```python
# Informasi dasar dataset
print(f"📊 Bentuk Dataset: {df.shape}")
print(f"   - Jumlah baris (light novel): {df.shape[0]}")  # 1366
print(f"   - Jumlah kolom (fitur): {df.shape[1]}")        # 4
```

**Variabel-variabel pada dataset adalah sebagai berikut:**

- **titles**: Judul lengkap light novel (object) - 1.366 judul unik
- **descriptions**: Deskripsi atau sinopsis novel yang menjelaskan plot dan karakter (object) - teks bebas dengan rata-rata 664 karakter
- **genres**: Genre novel dalam format list string (object) - 24 genre unik dengan Fantasy sebagai genre dominan (1.311 novel)
- **links**: URL referensi ke sumber informasi novel (object) - 1.366 link unik

**Exploratory Data Analysis (EDA):**

Analisis eksplorasi data mengungkap beberapa insight penting:

1. **Distribusi Genre:**

```python
# Hitung frekuensi setiap genre
all_genres = []
for genres in df['genres_list']:
    all_genres.extend(genres)

genre_counts = Counter(all_genres)
print(f"Total genre unik: {len(genre_counts)}")
print(f"Top 10 genre terpopuler:")
for genre, count in genre_counts.most_common(10):
    print(f"  {genre}: {count} novel")
```

- Fantasy mendominasi dengan 1.311 novel (96% dari total dataset)
- Kombinasi populer: Fantasy-Action-Adventure mencerminkan tren cerita petualangan heroik
- Terdapat 25 genre unik yang memberikan variasi cukup untuk sistem rekomendasi

![Analisis Distribusi Genre](/images/distribusi_genre.png)

2. **Analisis Panjang Deskripsi:**

```python
# Analisis panjang deskripsi
df['description_length'] = df['descriptions'].str.len()
print(f"Rata-rata panjang deskripsi: {df['description_length'].mean():.0f} karakter")
print(f"Deskripsi terpendek: {df['description_length'].min()} karakter")
print(f"Deskripsi terpanjang: {df['description_length'].max()} karakter")
```

- Rata-rata panjang deskripsi: 664 karakter
- Range: 26-3.037 karakter (distribusi right-skewed)
- Mayoritas deskripsi singkat dan padat, optimal untuk pemrosesan teks

![Analisis Panjang Deskripsi](/images/panjang_deskripsi.png)

**Kualitas Data:**

- ✅ Tidak ada missing values dalam dataset
- ✅ Semua novel memiliki informasi lengkap
- ✅ Format data konsisten dan siap untuk preprocessing

## Data Preparation

Tahapan data preparation dilakukan untuk mempersiapkan data agar dapat diproses oleh algoritma machine learning dengan optimal. Berikut adalah teknik-teknik yang diterapkan:

### 1. Text Preprocessing

**Proses yang dilakukan:**

```python
def clean_text(text):
    text = text.lower()                    # Convert ke lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus karakter khusus
    text = ' '.join(text.split())          # Hapus spasi berlebih
    return text

df['clean_descriptions'] = df['descriptions'].apply(clean_text)
```

**Alasan:** Text preprocessing diperlukan untuk menstandarkan format teks, menghilangkan noise (tanda baca, angka), dan memastikan konsistensi input untuk algoritma TF-IDF. Preprocessing yang tidak terlalu agresif mempertahankan konteks semantik yang penting.

### 2. Genre Processing

**Proses yang dilakukan:**

```python
def parse_genres(genre_str):
    try:
        return ast.literal_eval(genre_str)  # Parse string ke list
    except:
        return []

df['genres_list'] = df['genres'].apply(parse_genres)
df['genres_str'] = df['genres_list'].apply(lambda x: ' '.join(x).lower())
```

**Alasan:** Genre dalam format string list perlu di-parse untuk memudahkan pemrosesan. Konversi ke string tunggal memungkinkan integrasi dengan fitur teks lainnya dalam TF-IDF vectorization.

### 3. Feature Engineering

**Proses yang dilakukan:**

```python
df['combined_features'] = (
    df['clean_descriptions'] + ' ' +
    df['genres_str'] + ' ' +
    df['titles'].str.lower()
)
```

**Alasan:** Menggabungkan semua fitur tekstual (deskripsi, genre, judul) memberikan representasi holistik setiap novel. Pendekatan ini memungkinkan sistem menangkap kesamaan dari berbagai aspek novel, tidak hanya dari satu fitur saja.

### 4. Length Analysis

**Proses yang dilakukan:**

```python
df['description_length'] = df['descriptions'].str.len()
df['combined_length'] = df['combined_features'].str.len()
# Rata-rata combined features: 726 karakter
```

**Alasan:** Analisis panjang teks membantu memahami distribusi data dan memastikan tidak ada teks yang terlalu pendek atau panjang yang dapat mempengaruhi kualitas feature extraction.

**Hasil Preprocessing:**

- ✅ 1.366 novel berhasil diproses
- ✅ Combined features rata-rata 726 karakter
- ✅ Format data konsisten untuk tahap modeling

### 5. TF-IDF Vectorization

**Proses yang dilakukan:**

```python
# Inisialisasi TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,      # Batasi fitur untuk efisiensi
    stop_words='english',   # Hapus stop words
    ngram_range=(1, 2),     # Unigram dan bigram
    min_df=2,               # Minimum document frequency
    max_df=0.95             # Maximum document frequency
)

# Fit dan transform combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
# Output: Shape (1366, 5000) dengan sparsity 98.92%
```

TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk mengkonversi teks gabungan menjadi representasi numerik yang dapat diproses oleh algoritma machine learning. Teknik ini menganalisis seberapa penting suatu kata dalam dokumen relatif terhadap seluruh koleksi dokumen.

**Alasan:** TF-IDF diperlukan untuk mengkonversi data tekstual menjadi format numerik. Pendekatan ini memberikan representasi yang lebih bermakna dibandingkan simple word counting karena mempertimbangkan kepentingan relatif setiap kata.

## Modeling

Tahapan modeling mengimplementasikan sistem rekomendasi content-based filtering menggunakan algoritma Cosine Similarity untuk menyelesaikan permasalahan pencarian light novel yang relevan. Model dibangun berdasarkan representasi TF-IDF yang telah dihasilkan pada tahap Data Preparation.

### Content-Based Filtering dengan Cosine Similarity

**Cara Kerja Algoritma:**

Cosine Similarity adalah metrik yang mengukur kesamaan antara dua vektor berdasarkan cosinus sudut yang terbentuk di antara keduanya dalam ruang multidimensi. Dalam konteks sistem rekomendasi, setiap novel direpresentasikan sebagai vektor TF-IDF dalam ruang fitur 5000 dimensi. Algoritma menghitung similarity score antara semua pasangan novel dengan nilai berkisar 0-1, dimana 1 menunjukkan kesamaan sempurna dan 0 menunjukkan tidak ada kesamaan.

**Implementasi Model:**

```python
# Hitung cosine similarity matrix dari TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Output: Matrix 1366x1366 untuk perbandingan semua novel
```

Proses ini menghasilkan similarity matrix berukuran 1366x1366 yang berisi skor kesamaan antara setiap pasangan novel dalam dataset. Matrix ini menjadi dasar untuk mengidentifikasi novel-novel yang memiliki karakteristik serupa berdasarkan konten tekstual mereka.

### Fungsi Rekomendasi

Sistem rekomendasi dirancang untuk memberikan top-N recommendations berdasarkan similarity score tertinggi. Proses dimulai dengan mencari index novel referensi, kemudian mengambil similarity scores dari matrix, mengurutkan berdasarkan score tertinggi, dan menghasilkan daftar rekomendasi yang terstruktur.

```python
def get_content_recommendations(title, cosine_sim=cosine_sim, df=df, top_n=10):
    # Cari index novel dan hitung similarity scores
    idx = title_to_idx[title]
    sim_scores = cosine_sim[idx]

    # Sort dan ambil top N rekomendasi
    sim_scores_with_idx = sorted(enumerate(sim_scores),
                                key=lambda x: x[1], reverse=True)
    top_similar = sim_scores_with_idx[1:top_n+1]

    return recommendations_df
```

### Contoh Output Rekomendasi

**Input:** "Full Metal Panic!"

**Top 5 Rekomendasi:**

1. **Shinka no Mi** (Similarity: 0.2216)

   - Genre: Action, Adventure, Fantasy, Harem, Romance, School Life
   - Match reason: Kesamaan genre dan setting sekolah

2. **An Illustrated Guidebook to Other Races** (Similarity: 0.2160)
   - Genre: Action, Adventure, Fantasy, Harem, Romance, School Life
   - Match reason: Profil genre yang identik

**Kelebihan Pendekatan:**

- ✅ Tidak memerlukan data historis pengguna
- ✅ Dapat memberikan rekomendasi untuk item baru
- ✅ Transparansi dalam alasan rekomendasi
- ✅ Scalable untuk dataset besar

**Kelemahan Pendekatan:**

- ❌ Terbatas pada fitur konten yang tersedia
- ❌ Tidak menangkap preferensi pengguna yang dinamis
- ❌ Berpotensi menghasilkan rekomendasi yang terlalu mirip

## Evaluation

Evaluasi model dilakukan menggunakan multiple metrics untuk mengukur kualitas rekomendasi dari perspektif relevance, precision, dan novelty. Metrik evaluasi disesuaikan dengan konteks sistem rekomendasi dan business understanding yang telah ditetapkan.

### Metrik Evaluasi

#### 1. Genre Similarity Score

**Formula:**

```
Jaccard Similarity = |Intersection(Genre_A, Genre_B)| / |Union(Genre_A, Genre_B)|
```

**Cara Kerja:** Mengukur proporsi genre yang sama antara novel referensi dan novel yang direkomendasikan menggunakan Jaccard similarity coefficient.

#### 2. Precision at K

**Formula:**

```
Precision@K = (Jumlah rekomendasi relevan dalam top-K) / K
```

**Cara Kerja:** Mengukur proporsi rekomendasi yang memiliki minimal satu genre yang sama dengan novel referensi.

#### 3. Novelty Score

**Formula:**

```
Novelty = 1 - Average_Genre_Similarity
```

**Cara Kerja:** Mengukur seberapa berbeda rekomendasi dari novel referensi, memberikan indikator diversity.

### Hasil Evaluasi

**Evaluasi pada 10 Novel Sample:**

```python
# Hasil rata-rata metrik evaluasi
Average Genre Similarity: 0.6722  # 67.22% kesamaan genre
Average Precision: 0.9000         # 90% rekomendasi relevan
Average Novelty: 0.3278           # 32.78% kebaruan
```

![Rata-Rata Metrik Evaluasi](/images/hasil_evaluasi.png)

**Analisis Detail:**

1. **Genre Similarity (0.6722)**:

   - Skor tinggi menunjukkan model berhasil menemukan novel dengan genre serupa
   - Konsistensi baik dengan sebagian besar novel mendapat similarity 0.6-0.9
   - Novel "Watashi wa Teki ni Narimasen!" mencapai similarity tertinggi (0.936)

2. **Precision (0.9000)**:

   - 90% rekomendasi memiliki minimal 1 genre yang sama dengan referensi
   - 9 dari 10 novel test mendapat precision sempurna (1.0)
   - Menunjukkan akurasi tinggi dalam relevansi rekomendasi

3. **Novelty (0.3278)**:
   - Model memberikan keseimbangan antara relevance dan diversity
   - Tidak terjebak dalam "filter bubble" yang terlalu ketat
   - Variasi cukup untuk menghindari monotoni rekomendasi

### Hubungan dengan Business Understanding

**Problem Statement 1**: Information Overload

- ✅ **Teratasi**: Sistem berhasil menyaring 1.366 novel menjadi 10 rekomendasi teratas dengan precision 90%
- ✅ **Impact**: Mengurangi waktu pencarian dari manual browsing menjadi instant recommendation

**Problem Statement 2**: Kurangnya Personalisasi

- ✅ **Teratasi**: Content-based approach memberikan rekomendasi berdasarkan karakteristik spesifik novel yang diminati
- ✅ **Impact**: Setiap input menghasilkan rekomendasi yang unik dan personal

**Problem Statement 3**: Discovery Challenge

- ✅ **Teratasi**: Novelty score 32.78% menunjukkan sistem memberikan variasi yang cukup untuk discovery
- ✅ **Impact**: Pembaca dapat menemukan novel baru yang relevan namun tidak obvious

### Goals Achievement

**Goal 1**: Sistem Rekomendasi Otomatis

- ✅ **Tercapai**: Model dapat memberikan rekomendasi otomatis dengan response time < 1 detik
- ✅ **Metrik**: Precision 90% menunjukkan kualitas rekomendasi yang tinggi

**Goal 2**: Pengalaman Personalisasi

- ✅ **Tercapai**: Content-based filtering menganalisis deskripsi, genre, dan judul untuk personalisasi
- ✅ **Metrik**: Genre similarity 67.22% menunjukkan relevansi yang kuat

**Goal 3**: Peningkatan Discoverability

- ✅ **Tercapai**: Balance antara similarity (67.22%) dan novelty (32.78%)
- ✅ **Metrik**: Distribusi rekomendasi menunjukkan variasi yang sehat

### Solution Impact Analysis

**Content-Based Filtering dengan TF-IDF:**

- ✅ **Berdampak Positif**: Berhasil mencapai precision 90% dengan processing time efisien
- ✅ **Scalability**: Model dapat handle 1.366 novel dengan matrix 1366x1366 secara efisien
- ✅ **Maintainability**: Sistem dapat di-update dengan novel baru tanpa perlu retrain from scratch

**Technical Performance:**

- Model size: 17.29 MB (efisien untuk deployment)
- Sparsity: 98.92% (optimal memory usage)
- Coverage: 100% novel dapat direkomendasikan

**Kesimpulan Evaluasi:**

Sistem rekomendasi light novel telah berhasil memenuhi semua problem statements dan goals yang ditetapkan. Dengan precision 90%, genre similarity 67.22%, dan novelty 32.78%, model menunjukkan performa yang excellent dalam memberikan rekomendasi yang relevan, personal, dan diverse. Implementasi content-based filtering terbukti efektif untuk domain light novel dengan karakteristik konten yang kaya akan informasi tekstual.

---
