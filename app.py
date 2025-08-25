import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

# Load directly from TF Hub (downloads once and caches locally)
interpreter = tf.lite.Interpreter(model_path="cassava_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_from_model(input_data):
    """
    Function to run inference on the TFLite model.
    The input_data must be a preprocessed numpy array.
    """
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the image to match the model's input requirements.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define the class names list
class_names = [
    'Cassava Bacterial Blight (CBB)', 
    'Cassava Brown Streak Disease (CBSD)', 
    'Cassava Green Mottle (CGM)', 
    'Cassava Mosaic Disease (CMD)', 
    'Healthy'
]

disease_suggestions = {
    'Cassava Bacterial Blight (CBB)': "Apply copper-based fungicides, use resistant cassava varieties, and practice crop rotation to reduce bacterial spread.",
    'Cassava Brown Streak Disease (CBSD)': "Use disease-free planting materials, practice field sanitation, and remove and burn infected plants.",
    'Cassava Green Mottle (CGM)': "Ensure proper weed management, monitor insect vectors, and use resistant cassava cultivars.",
    'Cassava Mosaic Disease (CMD)': "Use certified virus-free cuttings, control whiteflies, and avoid intercropping with susceptible plants.",
    'Healthy': "Your cassava leaf looks healthy! Continue regular monitoring and apply good agricultural practices to keep it disease-free."
}

disease_colors = {
    'Cassava Bacterial Blight (CBB)': "danger",   # red
    'Cassava Brown Streak Disease (CBSD)': "danger",
    'Cassava Green Mottle (CGM)': "warning",     # yellow
    'Cassava Mosaic Disease (CMD)': "danger",
    'Healthy': "success"                         # green
}

# Create the 'images' directory if it doesn't exist
upload_folder = "static/uploads"
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)


@app.route("/", methods=['GET'] )
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' not in request.files or request.files['imagefile'].filename == '':
        return render_template('index.html', prediction="No image file selected.")

    imagefile = request.files['imagefile']
    image_path = os.path.join(upload_folder, imagefile.filename)
    imagefile.save(image_path)
    

    # Get the required input dimensions for the model
    # Most image models have shape [1, height, width, 3]
    target_height, target_width = input_details[0]['shape'][1:3]

    # Preprocess the uploaded image using the new function
    preprocessed_input = preprocess_image(image_path, target_size=(target_height, target_width))

    # Get predictions
    predictions = predict_from_model(preprocessed_input)

    predicted_class_index = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100

    # Use the predicted index to get the class name
    if 0 <= predicted_class_index < len(class_names):
        predicted_class_name = class_names[predicted_class_index]
    else:
        predicted_class_name = "Unknown disease"

    
    suggestion = disease_suggestions.get(predicted_class_name, "No suggestion available.")
    color = disease_colors.get(predicted_class_name, "secondary")
    
    # Pass the prediction string directly to the template
    return render_template(
        'index.html',
        prediction=predicted_class_name,
        suggestion=suggestion,
        color=color,
        confidence=round(confidence, 2),
        image=imagefile.filename
    )

if __name__ == '__main__':
    app.run(debug=True)