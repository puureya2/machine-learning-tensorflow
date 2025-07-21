# 🌾 Crops Sorter Model

A machine learning pipeline for detecting crop health patterns using **Convolutional Neural Networks (CNNs)**. The system is trained on 10,000+ labeled agricultural images to predict crop conditions with high accuracy, and includes analytics and visualizations of model performance.

---

## 🧠 Core Features

- 🤖 Trained 3 custom CNN architectures using **TensorFlow** and **Keras**
- 📊 Achieved **90.07% accuracy** on validation data
- 📈 Designed analytics scripts to evaluate 4 key model metrics:
  - Accuracy
  - Precision
  - Recall
  - Loss
- 🌐 Visualized predictions using **Matplotlib** with bar graphs and cluster charts

---

## 🛠️ Tech Stack

| Purpose                 | Technology         |
|-------------------------|--------------------|
| Model Training          | TensorFlow, Keras  |
| Data Preprocessing      | Scikit-learn       |
| Data Visualization      | Matplotlib         |
| Language                | Python             |

---

## 🧪 Training Overview

- Dataset: >10,000 images across various crop-health categories  
- Preprocessing: Normalization, augmentation, and one-hot encoding  
- CNN Variants:
  - ResNet-inspired shallow net
  - Custom-built 6-layer CNN
  - Lightweight MobileNetv2 baseline  
- Evaluation: Accuracy calculated via validation split (90.07%)

---

## 📊 Visualizations

Using `matplotlib`, the following charts were generated:
- **Bar graphs** comparing model performance
- **Cluster plots** to group prediction categories
- Accuracy/Loss trends over training epochs

---

## 📁 Files

- `train_model.py` — Model architecture and training loop
- `evaluate.py` — Script to calculate and compare model metrics
- `visualize.py` — Generates performance graphs
- `README.md` — Project documentation

---

## 🧠 Future Improvements

- Integrate early stopping and learning rate schedulers
- Expand dataset with underrepresented crops
- Deploy trained model to a web/mobile interface for farmer use

---

## 📅 Timeline

**January – February 2025**  
Created as a solo AI-agriculture capstone to explore the intersection of deep learning and food security.

---

## 📬 Contact

**Kevin Chifamba**  
📧 kevinnanashe@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/yourprofile) • [GitHub](https://github.com/your-username)

---
