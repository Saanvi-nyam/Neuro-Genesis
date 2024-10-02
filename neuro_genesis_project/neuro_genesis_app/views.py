from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .utils import predict_tumour  # Import the new function
from django.conf import settings
from keras.models import load_model
import os
import openai  # Import OpenAI library

# Load the pre-trained model
model = load_model(os.path.join(settings.BASE_DIR, 'bestmodel.h5'))

# Use environment variable to set your OpenAI API key
openai.api_key = "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"



def generate_openai_response(prompt):
    """Function to generate a response from the OpenAI API."""
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',  # Use a supported model
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return str(e)  # Return error message

def index(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']  # Get uploaded file
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)  # Save file
        file_url = fs.url(file_path)  # Generate the URL for the file

        # Make prediction using the model
        result = predict_tumour(model, os.path.join(settings.MEDIA_ROOT, file_path))

        # Generate an AI response using OpenAI API based on the prediction result
        prompt = f"Explain the results for a brain tumor prediction: {result}"
        ai_response = generate_openai_response(prompt)

        # Render the template with the prediction result and the file URL
        return render(request, 'neuro_genesis_app/index.html', {
            'result': result,
            'uploaded_file': file_url,  # Pass the file URL to the template
            'ai_response': ai_response  # Pass the AI-generated response to the template
        })

    # Render the page with an empty form
    return render(request, 'neuro_genesis_app/index.html')
