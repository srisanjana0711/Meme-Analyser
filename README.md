

# Meme Analyser  

## 📌 Overview  
Meme Analyser is an AI-powered tool that detects and classifies memes based on their content, sentiment, and context. It utilizes deep learning models for image and text analysis, extracting meaningful insights from memes shared on social media.  

## 🚀 Features  
-  **Image Recognition** – Identifies objects, text, and elements in memes.  
-  **Text Extraction (OCR)** – Extracts embedded text from memes for analysis.  
-  **Sentiment Analysis** – Determines whether the meme is positive, negative, or neutral.  
-  **Meme Classification** – Categorizes memes into predefined themes (e.g., political, humor, relatable, etc.).  
-  **AI-Powered Understanding** – Uses NLP and vision models to interpret meme meaning.  

## 🛠 Tech Stack  
- **Python** – Core programming language  
- **TensorFlow / PyTorch** – Deep learning models  
- **CLIP (by OpenAI)** – Image and text similarity analysis  
- **Tesseract OCR** – Text extraction from memes  
- **Transformers (Hugging Face)** – NLP-based sentiment analysis  
- **Gradio** – For building an interactive UI  

## 📂 Project Structure  
```
Meme-Analyser/
│── meme_analyzer (1).ipynb  # Jupyter Notebook containing the complete implementation
│── README.md                # Project documentation
```

## ⚡ Installation  

### 1️⃣ Run on Google Colab  
To set up the environment in **Google Colab**, run the following commands:  
```python
!pip install git+https://github.com/openai/CLIP.git
!pip install torch torchvision torchaudio
!pip install pytesseract
!pip install transformers
!pip install gradio
!pip install easyocr
!pip install clip-by-openai
!pip install numpy
!pip install pillow
!pip install requests
!pip install opencv-python
```

### 2️⃣ Open the Jupyter Notebook  
Upload **`meme_analyzer (1).ipynb`** to Google Colab and run the cells sequentially.  

### 3️⃣ Set Up Hugging Face API Key  
Make sure to replace `**KEY**` in the notebook code with your **Hugging Face API Key** to access the models.  
```python
HUGGINGFACE_API_KEY = "your_api_key_here"
```
>  **Note:** You can get a free Hugging Face API key from [Hugging Face](https://huggingface.co/settings/tokens).  

## 📸 Output Screenshots  


<img width="791" alt="image" src="https://github.com/user-attachments/assets/b6d42506-f4da-42fc-a9c8-31aec3522fca" />

### Negative meme 
<img width="804" alt="image" src="https://github.com/user-attachments/assets/5f192d61-d99f-4a29-8abe-600b20fcaa12" />

### Positive meme 
<img width="799" alt="image" src="https://github.com/user-attachments/assets/ec8b606b-57b5-4761-b7dc-14eaff936029" />


### **🧑‍💻 Author**
👤 **Srisanjana Karunamoorthy**  
🔗 GitHub: [srisanjana0711](https://github.com/srisanjana0711)  

---
