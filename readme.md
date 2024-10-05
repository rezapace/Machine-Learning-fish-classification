# Fish Classification

# HOW TO RUN PROJECT

1. Download the Jupyter notebook:
   ```bash
   wget https://github.com/rezapace/Machine-Learning-fish-classification/releases/download/1.0/fish_classification.ipynb
   ```

2. Open Google Colab:
   [https://colab.research.google.com/#create=true](https://colab.research.google.com/#create=true)

3. Upload the downloaded `klasifikasi_catur.ipynb` file to Google Colab.

4. Run the notebook:
   - Execute each cell in order
   - Follow the instructions provided in the notebook comments

Note: Make sure you have a Google account to use Google Colab. If you encounter any issues, please refer to the [project repository](https://github.com/rezapace/Machine-Learning-fish-classification) for troubleshooting or to report problems.


## Deskripsi
Proyek ini bertujuan untuk mengklasifikasikan jenis ikan menggunakan model pembelajaran mesin berbasis gambar. Dataset yang digunakan terdiri dari gambar-gambar ikan yang telah dikelompokkan ke dalam beberapa kategori. Model yang digunakan dalam proyek ini adalah VGG16 dan ResNet50, yang merupakan model deep learning populer untuk tugas klasifikasi gambar.

## Kegunaan
Proyek ini dapat digunakan untuk:
- Mengidentifikasi jenis ikan dari gambar.
- Membandingkan performa dua model deep learning (VGG16 dan ResNet50) dalam tugas klasifikasi gambar.
- Mempelajari teknik augmentasi data dan evaluasi model menggunakan metrik seperti confusion matrix dan classification report.

## Fungsi
Proyek ini memiliki beberapa fungsi utama:
1. **create_data_generators**: Membuat generator data untuk pelatihan, validasi, dan pengujian.
2. **evaluate_model**: Mengevaluasi performa model menggunakan confusion matrix dan classification report.

## Bagaimana Menjalankan
1. **Instalasi Dependensi**:
   Pastikan Anda memiliki Jupyter Notebook atau Google Colab untuk menjalankan file `.ipynb` ini. Instal dependensi yang diperlukan dengan menjalankan perintah berikut di sel kode pertama:
   ```python
   !apt install unzip
   !unzip -q "/content/Dataset.zip"
   !apt install git
   !git clone "https://github.com/rezapace/Machine-Learning-fish-classification"
   ```

2. **Import Library**:
   Import library yang diperlukan dengan menjalankan sel kode berikut:
   ```python
   import tensorflow as tf
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   from tensorflow.keras.applications import VGG16, ResNet50
   from tensorflow.keras.layers import GlobalMaxPooling2D, Dense
   from tensorflow.keras.models import Sequential, load_model
   from tensorflow.keras.callbacks import ModelCheckpoint
   from sklearn.metrics import confusion_matrix, classification_report
   ```

3. **Membuat Data Generators**:
   Buat generator data untuk pelatihan, validasi, dan pengujian dengan menjalankan sel kode berikut:
   ```python
   def create_data_generators(train_dir, valid_dir, test_dir, target_size=(224, 224), batch_size=32, preprocessing_function=None):
       train_datagen = ImageDataGenerator(
           rotation_range=40,
           horizontal_flip=True,
           preprocessing_function=preprocessing_function
       )
       valid_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
       test_datagen = ImageDataGenerator()

       train_generator = train_datagen.flow_from_directory(
           train_dir, target_size=target_size, batch_size=batch_size,
           class_mode='categorical', seed=42, shuffle=True
       )
       valid_generator = valid_datagen.flow_from_directory(
           valid_dir, target_size=target_size, batch_size=batch_size,
           class_mode='categorical', seed=42, shuffle=False
       )
       test_generator = test_datagen.flow_from_directory(
           test_dir, target_size=target_size, batch_size=batch_size,
           class_mode='categorical', seed=42, shuffle=False
       )

       return train_generator, valid_generator, test_generator

   train_dir = "/content/Dataset/Data_Train"
   valid_dir = "/content/Dataset/Data_Validasi"
   test_dir = "/content/Dataset/Data_Uji"

   vgg16_train, vgg16_valid, vgg16_test = create_data_generators(
       train_dir, valid_dir, test_dir,
       preprocessing_function=tf.keras.applications.vgg16.preprocess_input
   )
   resnet50_train, resnet50_valid, resnet50_test = create_data_generators(
       train_dir, valid_dir, test_dir,
       preprocessing_function=tf.keras.applications.resnet50.preprocess_input
   )
   ```

4. **Evaluasi Model**:
   Evaluasi performa model dengan menjalankan sel kode berikut:
   ```python
   def evaluate_model(model, test_data, class_labels):
       predictions = model.predict(test_data)
       predicted_classes = np.argmax(predictions, axis=1)
       true_classes = test_data.classes

       cm = confusion_matrix(true_classes, predicted_classes)
       plt.figure(figsize=(8, 6))
       sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
       plt.xlabel('Predicted')
       plt.ylabel('Actual')
       plt.title('Confusion Matrix')
       plt.show()

       report = classification_report(true_classes, predicted_classes, target_names=class_labels)
       print(report)

   class_labels = list(vgg16_test.class_indices.keys())

   print("VGG16 Model Evaluation:")
   evaluate_model(vgg16_model, vgg16_test, class_labels)

   print("ResNet50 Model Evaluation:")
   evaluate_model(resnet50_model, resnet50_test, class_labels)
   ```

5. **Menyimpan Model**:
   Simpan model yang telah dilatih dengan menjalankan sel kode berikut:
   ```python
   vgg16_model.save("/content/model_vgg16_GlobalMax_aug.h5")
   resnet50_model.save("/content/model_resnet50_GlobalMax_aug.h5")
   ```

## Kesimpulan
Proyek ini berhasil mengimplementasikan dua model deep learning, VGG16 dan ResNet50, untuk tugas klasifikasi gambar ikan. Dengan menggunakan teknik augmentasi data dan evaluasi model yang tepat, proyek ini menunjukkan bagaimana model deep learning dapat digunakan untuk mengklasifikasikan gambar dengan akurasi yang baik. Hasil evaluasi menunjukkan bahwa kedua model memiliki performa yang memuaskan dalam mengklasifikasikan jenis ikan dari gambar.