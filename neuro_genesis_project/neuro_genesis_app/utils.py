import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import openai  # Import OpenAI to interact with its API
from django.conf import settings  # Import settings to access OpenAI API key

# Predict if the image contains a tumor
def predict_tumour(model, image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    input_arr = img_to_array(image) / 255
    input_array = np.expand_dims(input_arr, axis=0)  # Add batch dimension

    # Use the loaded model to make predictions
    pred = model.predict(input_array)
    
    # Interpretation of the prediction
    if pred >= 0.5:
        return "The Image Does Not Contain A Tumour"
    else:
        return "The Image Does Contain A Tumour"


# Generate a text explanation using OpenAI API
def generate_openai_response(prompt):
    openai.api_key = settings.OPENAI_API_KEY
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",  # Model used for text generation
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()
