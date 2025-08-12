import gradio as gr
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. Load FAISS Index and Metadata
print("Loading FAISS index...")
index = faiss.read_index("fitness_index.faiss")  # Load FAISS index
metadata = pd.read_csv("fitness_metadata.csv")  # Load metadata file

# 2. Load Sentence Transformer Model
print("Loading Sentence Transformer model...")
embedding_model = SentenceTransformer("fitness_model_files")  # Replace with path to Sentence Transformer folder

# 3. Load Vision Transformer Model
print("Loading Vision Transformer model...")
try:
    vit_model = load_model("vit_workout_model.keras")  # Vision Transformer model
    # Define class names (ensure these match the ViT training labels)
    class_names = [
        "barbell biceps curl", "bench press", "chest fly machine", "deadlift", "decline bench press", "hammer curl", "hip thrust", 
        "incline bench press", "lat pulldown", "lateral raises", "leg extension", "leg raises", "plank", "pull up", "push up", "romanian deadlift",
        "russian twist", "shoulder press", "squat", "t bar row", "tricep dips", "tricep pushdown"
    ]
    print("Vision Transformer model loaded successfully.")
except Exception as e:
    vit_model = None
    class_names = []
    print(f"Vision Transformer model not available. Reason: {e}")

# 4. Retrieve Exercises from FAISS
def retrieve_exercises(query, k=5):
    """
    Retrieve the top-k exercises based on the user query.
    """
    # Generate query embedding
    query_embedding = embedding_model.encode([query])

    # Search the FAISS index
    distances, indices = index.search(query_embedding, k)

    # Retrieve metadata for top results
    results = []
    for idx in indices[0]:
        results.append(metadata.iloc[idx]["content"])

    return results

# 5. Image Classification Function
def classify_image(img):
    """
    Classify an uploaded image using the Vision Transformer model.
    """
    if not vit_model:
        return "Vision Transformer model not available."

    # Preprocess the image
    img = img.resize((224, 224))  # Resize to match ViT input dimensions
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values

    # Predict the workout type
    predictions = vit_model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    return f"Predicted Workout: {predicted_class} (Confidence: {confidence:.2f}%)"

# 6. Define the Chatbot Function
def chatbot(query, img=None):
    """
    Chatbot function to respond to user queries and classify images.
    """
    response = ""

    # Text-based workout suggestions
    if query:
        results = retrieve_exercises(query, k=5)
        response += "Here are some workout suggestions based on your query:\n\n"
        for i, result in enumerate(results, 1):
            response += f"{i}. {result}\n\n"

    # Image classification
    if img:
        classification_result = classify_image(img)
        response += f"\n{classification_result}\n"

    return response

# 7. Build the Gradio Interface
interface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Textbox(lines=2, placeholder="Ask me about workouts! E.g., 'Beginner yoga exercises'"),
        gr.Image(type="pil", label="Upload a workout image (optional)")
    ],
    outputs="text",
    title="Fitness Chatbot with Image Classification",
    description="Ask the chatbot about fitness exercises or upload an image to classify a workout."
)

# 8. Launch the Interface
if __name__ == "__main__":
    interface.launch()
