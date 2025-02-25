# 🌪️ Disaster Tweet Classification Using Generative AI

This project focuses on classifying tweets as **disaster-related** or **non-disaster-related** using **Generative AI techniques**. The goal is to help disaster relief organizations and news agencies filter critical information from social media during emergencies.

![Disaster Classification](https://img.shields.io/badge/NLP-GenerativeAI-blue) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📁 **Project Structure**

In this project:

1. **`Disaster_Tweet_Classification.ipynb`**  
   - The main notebook where the project was executed, including both successful runs and error handling.

2. **`disaster_tweet_classification.py`**  
   - A pure Python script version of the project, without notebook cells or markdown.

3. **`Disaster_Tweet_Classification_CompleteCode.ipynb`**  
   - A clean version of the notebook containing only the successfully executed code cells (no errors or failed attempts).

4. **`train.csv`** & **`test.csv`**  
   - The datasets used for training and testing the model:
     - **`train.csv`** → Used to train the disaster tweet classification model.
     - **`test.csv`** → Used to evaluate and test the trained model.

---

This structure gives users a clear understanding of the files and their purposes.

💡 **Tip:** You can include this under a **"Project Structure"** or **"File Descriptions"** section in your `README.md`.  

Would you like help integrating this into the full **README.md** file? 🚀




---

## 📌 **Table of Contents**
- [💡 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🛠️ Technologies Used](#️-technologies-used)
- [⚙️ Installation](#️-installation)
- [🚀 How to Run the Project](#-how-to-run-the-project)
- [📈 Results](#-results)
- [🤔 Future Improvements](#-future-improvements)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 💡 **Project Overview**
In emergency situations, platforms like **Twitter** become a primary source for real-time information. However, distinguishing between relevant disaster-related tweets and unrelated content is challenging for machines. 

This project uses **Natural Language Processing (NLP)** and **Generative AI** techniques like:
- **Text Preprocessing** using **NLTK** 🧹
- **Word Embeddings** using **Word2Vec** 📖
- **Zero-Shot Classification** using **Hugging Face's BART model** 🤖

The primary objective is to create a pipeline that classifies tweets as **"disaster"** or **"non-disaster"** with minimal human intervention.

---

## 📊 **Dataset**
The project uses the **[Kaggle - NLP Getting Started Dataset](https://www.kaggle.com/competitions/nlp-getting-started)**.

**Dataset Features:**
- `id` — Unique identifier for each tweet
- `text` — The tweet content
- `target` — `1` if disaster-related, `0` otherwise

---

## 🛠️ **Technologies Used**
- 🐍 **Python 3.x**  
- 📊 **Pandas** & **NumPy** — Data manipulation  
- 🧹 **NLTK** — Text preprocessing  
- 🧠 **Gensim (Word2Vec)** — Word embeddings  
- 🤗 **Hugging Face Transformers** — Zero-Shot classification  
- 📉 **scikit-learn** — Model evaluation  
- 📈 **Matplotlib & Seaborn** — Data visualization  
- 💻 **Google Colab** — Development environment

---

## ⚙️ **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Akashmk1803/Disaster-Tweet-Classification-Using-generative-AI.git
   cd Disaster-Tweet-Classification-Using-generative-AI

2. Install the required libraries:
    pip install pandas numpy nltk gensim transformers scikit-learn matplotlib tqdm

3. Download NLTK resources (run in Python):
      import nltk
      nltk.download('stopwords')
      nltk.download('punkt')
      nltk.download('wordnet')





