# 🏋️ AI-Powered Fitness Routine Planner

## 📌 Overview
The **AI-Powered Fitness Routine Planner** is a **Generative AI** project designed to create **personalized workout plans** based on user preferences, fitness levels, and goals.  
It integrates:
- **Large Language Models (LLMs)** for natural language understanding
- **FAISS** for semantic search over fitness data
- **Vision Transformer (ViT)** for workout image classification

The system enables users to receive **context-aware exercise recommendations** and identify workout types from uploaded images.

---

## 🎯 Features
- **💬 Text-based Workout Suggestions** – Fine-tuned sentence transformer model with FAISS index for relevant exercises.  
- **🖼️ Image-based Workout Classification** – Classifies workout images into 22 categories via Vision Transformer.  
- **🤖 Dynamic Chatbot Interface** – Built with **Gradio** for interactive user queries.  
- **📊 Dataset-driven Recommendations** – Curated fitness datasets from Kaggle ensure accuracy.  

---

## 🛠️ Tech Stack
**Languages:**
- Python  

**Libraries & Frameworks:**
- [FAISS](https://github.com/facebookresearch/faiss) – Similarity search  
- [Sentence Transformers](https://www.sbert.net/) – Text embeddings  
- [TensorFlow/Keras](https://www.tensorflow.org/) – Vision Transformer model  
- [Gradio](https://www.gradio.app/) – Interactive UI  
- Pandas, NumPy – Data processing  

**Datasets:**
- [Gym Exercises Dataset](https://www.kaggle.com/datasets/ambarishdeb/gym-exercises-dataset)  
- [Exercise and Fitness Metrics Dataset](https://www.kaggle.com/datasets/aakashjoshi123/exercise-and-fitness-metrics-dataset)  
- [Ultimate Gym Exercises Dataset](https://www.kaggle.com/datasets/peshimaammuzammil/the-ultimate-gym-exercises-dataset-for-all-levels)  
- [Gym Exercise Data](https://www.kaggle.com/datasets/niharika41298/gym-exercise-data)  

---

## 🚀 How It Works
1. **User Query** – Enter a workout-related query.  
2. **Semantic Search** – Sentence Transformer encodes query → FAISS retrieves top matches.  
3. **Optional Image Upload** – Vision Transformer predicts workout type.  
4. **Response Generation** – Gradio UI returns suggestions and classification.

---

## 📸 Preview
<img width="1893" height="915" alt="UI" src="https://github.com/user-attachments/assets/7fd1b906-0d55-40af-9cea-069a54d24b99" />

---

## 📄 References
Covered in Proposal.pdf:
- AI in fitness development
- ML-based workout recommendation systems
- AI-powered trainers & guides
- Pose estimation & real-time feedback
- GPT-4 in exercise prescription

---

## 👨‍💻 Authors
Aasir Farrukh – i210375@nu.edu.pk

Syed Muhammad Hassan Raza – i210465@nu.edu.pk

Moiez Asif – i212483@nu.edu.pk
