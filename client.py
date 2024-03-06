from flask import Flask, render_template, request, redirect, url_for
import requests
from tensorflow import keras
import numpy as np
import base64
from keras.utils import to_categorical
app = Flask(__name__)

# Placeholder for the client model
client_model = None
questions = [
    "Do you often feel nervous?",
    "Do you experience panic attacks?",
    "Do you experience rapid breathing?",
    "Do you often sweat excessively?",
    "Do you have trouble concentrating?",
    "Do you have trouble sleeping?",
    "Do you have trouble with work or daily tasks?",
    "Do you often feel hopeless?",
    "Do you experience frequent anger or irritability?",
    "Do you tend to overreact to situations?",
    "Have you noticed a change in your eating habits?",
    "Have you experienced suicidal thoughts?",
    "Do you often feel tired or fatigued?",
    "Do you have close friends you can confide in?",
    "Do you spend excessive time on social media?",
    "Have you experienced significant weight gain or loss?",
    "Do you place a high value on material possessions?",
    "Do you tend to keep to yourself or prefer solitude?",
    "Do you frequently experience distressing memories?",
    "Do you have nightmares frequently?",
    "Do you tend to avoid people or activities?",
    "Do you often feel negative about yourself or your life?",
    "Do you have trouble concentrating or focusing?",
    "Do you often blame yourself for things?"
]
def initialize_client_model():
    global client_model

    # Fetch model architecture and weights from the server
    response = requests.get('http://localhost:5000/get_model')
    model_data = response.json()

    # Reconstruct the model from JSON
    model = keras.models.model_from_json(model_data['model'])

    # Convert base64-encoded weights back to bytes
    response = requests.get('http://localhost:5000/get_weights')
    with open('model_weights.h5', 'wb') as f:
        f.write(response.content)

    # Load model weights
    model.load_weights('model_weights.h5')

    # Set the received model as the client model
    client_model = model

# Initialize the client model when the Flask app starts
initialize_client_model()
@app.route('/')
def index():
    return render_template('index.html', questions=questions)

# @app.route('/predict', methods=['POST'])
# def predict():
#     global client_model

#     input_data = request.json['data']

#     predictions = client_model.predict(input_data)

    
#     X_train = np.array(input_data)
#     y_train = np.array(predictions)  # Just an example, replace it with actual target labels
#     client_model.fit(X_train, y_train, epochs=1, verbose=0)

#     # Send the updated model weights to the server for aggregation
#     send_updated_model_to_server()

#     return 'Prediction made, model updated, and model updates sent to server successfully!'

@app.route('/get_model_from_server', methods=['GET'])
def get_model_from_server():
    global client_model

    # Fetch model architecture and weights from the server
    response = requests.get('http://localhost:5000/get_model')
    model_data = response.json()

    # Reconstruct the model from JSON
    model = keras.models.model_from_json(model_data['model'])

    # Convert base64-encoded weights back to bytes
    #weights_data = [base64.b64decode(w) for w in model_data['weights']]
    response = requests.get('http://localhost:5000/get_weights')

    with open('model_weights.h5', 'wb') as f:
        f.write(response.content)
        # Set model weights
    #weightss = response.content
    model.load_weights('model_weights.h5')
    new_data_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    new_data_array = np.array(new_data_list)

# Reshape the array to match the input shape expected by the model
    new_data_array = new_data_array.reshape(1, 24)
    answ = model.predict(new_data_array)
    # Set the received model as the client model
    client_model = model

    return str(answ)

def send_updated_model_to_server():
    global client_model

    # Extract the updated model weights
    updated_weights = client_model.get_weights()

    # Send the updated model weights to the server for aggregation
    requests.post('http://localhost:5000/update_model', json={'weights': updated_weights})

    return 'Updated model weights sent to server for aggregation!'

@app.route('/predict', methods=['POST'])
def predict():
    global client_model

    # Check if client_model is initialized
    if client_model is None:
        # Re-initialize the client model
        initialize_client_model()

    # Get answers from the request
    answers = [request.form.get(f'answer-{i}') for i in range(24)]

    # Convert answers to one-hot encoding
    encoded_answers = [1 if ans == 'yes' else 0 for ans in answers]
    input_data = np.array([encoded_answers])
    # Make prediction
    predictions = client_model.predict(input_data)

    # Convert prediction to disorder label
    disorder_labels = ['Anxiety', 'Depression', 'Stress', 'Normal', 'Loneliness']
    predicted_label = np.argmax(predictions)
    prediction_result = disorder_labels[predicted_label]
    update_client_model(encoded_answers, predicted_label)
    send_updated_model_to_server()
    # Pass questions and prediction result to the template
    return render_template('index.html', questions=questions, prediction=prediction_result, answers=answers)

import numpy as np

def update_client_model(input_data, new_target_label):
    global client_model
    if client_model is None:
        # Re-initialize the client model
        initialize_client_model()
    print("HEY GUYS:", input_data, new_target_label)
    # Convert input_data and new_target_label to NumPy arrays if they're not already
    new_target_label_one_hot = to_categorical(new_target_label, num_classes=5)
    input_data = np.array(input_data)
    #l = []
    new_target_label = np.array(new_target_label)
    #new_target_label.append(new_target_label)
    print("HEY GUYS1:", input_data.shape, new_target_label_one_hot)
    print("MKV:", client_model.summary())
    # Compile the model if it hasn't been compiled already
    if not client_model._is_compiled:
        client_model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Train the client model on the new data
    client_model.fit(input_data, new_target_label, epochs=1, batch_size=1, verbose=0)



def send_updated_model_to_server():
    global client_model

    # Extract the updated model weights
    updated_weights = client_model.get_weights()
    
    # Send the updated model weights to the server for aggregation
    requests.post('http://localhost:5000/update_model', json={'weights': updated_weights})

    return 'Updated model weights sent to server for aggregation!'


@app.route('/finish')
def finish():
    # Render the finish template
    return render_template('finish.html')

if __name__ == '__main__':  
    app.run(debug=True, port=5001)
