from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import requests
from tensorflow import keras
import numpy as np
import base64
import pandas as pd
import google.generativeai as genai
import os
from keras.utils import to_categorical
app = Flask(__name__)


genai.configure(api_key='AIzaSyB9FdmuacT7pk9jK6Hb25unKefYnb6EY54')
model = genai.GenerativeModel('models/gemini-pro')
# Placeholder for the client model
client_model = None
questions = [
    "How frequently do you feel nervous?",
    "How frequently do you experience panic attacks?",
    "How frequently do you experience rapid breathing?",
    "How frequently do you sweat excessively?",
    "How frequently do you have trouble concentrating?",
    "How frequently do you have trouble sleeping?",
    "How frequently do you have trouble with work or daily tasks?",
    "How frequently do you feel hopeless?",
    "How frequently do you experience frequent anger or irritability?",
    "How frequently do you tend to overreact to situations?",
    "Have you noticed a change in your eating habits?",
    "How frequently have you experienced suicidal thoughts?",
    "How frequently do you feel tired or fatigued?",
    "Do you have close friends you can confide in?",
    "Do you spend excessive time on social media?",
    "Have you experienced significant weight gain or loss?",
    "Do you place a high value on material possessions?",
    "Do you tend to keep to yourself or prefer solitude?",
    "Do you frequently experience distressing memories?",
    "How frequently do you have nightmares?",
    "Do you tend to avoid people or activities?",
    "How frequently do you feel negative about yourself or your life?",
    "Do you have trouble concentrating or focusing?",
    "How frequently do you blame yourself for things?"
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
    with open('model_weights.weights.h5', 'wb') as f:
    with open('model_weights.weights.h5', 'wb') as f:
        f.write(response.content)

    # Load model weights
    model.load_weights('model_weights.weights.h5')
    model.load_weights('model_weights.weights.h5')

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

    #weights_data = [base64.b64decode(w) for w in model_data['weights']]
    response = requests.get('http://localhost:5000/get_weights')

    with open('model_weights.weights.h5', 'wb') as f:
    with open('model_weights.weights.h5', 'wb') as f:
        f.write(response.content)
        # Set model weights
    #weightss = response.content
    model.load_weights('model_weights.weights.h5')
    model.load_weights('model_weights.weights.h5')
    new_data_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    new_data_array = np.array(new_data_list)

    new_data_array = new_data_array.reshape(1, 24)
    answ = model.predict(new_data_array)
    client_model = model

    return str(answ)

def send_updated_model_to_server():
    global client_model

    updated_weights = client_model.get_weights()

    requests.post('http://localhost:5000/update_model', json={'weights': updated_weights})

    return 'Updated model weights sent to server for aggregation!'


def get_recommendation_for_question(question):
    recommendations = {
        "How frequently do you feel nervous?": {
            "recommendation": "Practice mindfulness or deep breathing exercises.",
            "weights": {"Anxiety": 0.6, "Depression": 0.3, "Stress": 0.4, "Normal": 0.1, "Loneliness": 0.2}
        },
        "How frequently do you experience panic attacks?": {
            "recommendation": "Practice deep breathing exercises and mindfulness techniques to calm the nervous system.",
            "weights": {"Anxiety": 0.8, "Depression": 0.5, "Stress": 0.7, "Normal": 0.3, "Loneliness": 0.4}
        },
        "How frequently do you experience rapid breathing?": {
            "recommendation": "Try to calm yourself down by taking deep breaths.",
            "weights": {"Anxiety": 0.5, "Depression": 0.2, "Stress": 0.4, "Normal": 0.1, "Loneliness": 0.3}
        },
        "How frequently do you sweat excessively?": {
            "recommendation": "Consider practicing relaxation techniques to manage stress.",
            "weights": {"Anxiety": 0.4, "Depression": 0.3, "Stress": 0.5, "Normal": 0.2, "Loneliness": 0.3}
        },
        "How frequently do you have trouble concentrating?": {
            "recommendation": "Create a quiet and organized workspace to improve focus.",
            "weights": {"Anxiety": 0.2, "Depression": 0.1, "Stress": 0.3, "Normal": 0.1, "Loneliness": 0.2}
        },
        "How frequently do you have trouble sleeping?": {
            "recommendation": "Establish a regular sleep schedule and avoid caffeine before bedtime.",
            "weights": {"Anxiety": 0.4, "Depression": 0.3, "Stress": 0.5, "Normal": 0.2, "Loneliness": 0.3}
        },
        "How frequently do you have trouble with work or daily tasks?": {
            "recommendation": "Break tasks into smaller, manageable steps.",
            "weights": {"Anxiety": 0.3, "Depression": 0.2, "Stress": 0.4, "Normal": 0.1, "Loneliness": 0.2}
        },
        "How frequently do you feel hopeless?": {
            "recommendation": "Talk to a trusted friend or family member about your feelings.",
            "weights": {"Anxiety": 0.5, "Depression": 0.6, "Stress": 0.4, "Normal": 0.2, "Loneliness": 0.3}
        },
        "How frequently do you experience frequent anger or irritability?": {
            "recommendation": "Practice stress-reduction techniques like meditation or yoga.",
            "weights": {"Anxiety": 0.4, "Depression": 0.3, "Stress": 0.5, "Normal": 0.2, "Loneliness": 0.3}
        },
        "How frequently do you tend to overreact to situations?": {
            "recommendation": "Practice mindfulness and try to respond calmly to triggers.",
            "weights": {"Anxiety": 0.3, "Depression": 0.2, "Stress": 0.4, "Normal": 0.1, "Loneliness": 0.2}
        },
        "Have you noticed a change in your eating habits?": {
            "recommendation": "Consult a nutritionist or healthcare professional for guidance.",
            "weights": {"Anxiety": 0.2, "Depression": 0.1, "Stress": 0.3, "Normal": 0.1, "Loneliness": 0.2}
        },
        "How frequently have you experienced suicidal thoughts?": {
            "recommendation": "Seek immediate help from a mental health professional or hotline.",
            "weights": {"Anxiety": 0.9, "Depression": 0.8, "Stress": 0.7, "Normal": 0.4, "Loneliness": 0.6}
        },
        "How frequently do you feel tired or fatigued?": {
            "recommendation": "Prioritize sleep, exercise, and healthy eating to boost energy levels.",
            "weights": {"Anxiety": 0.4, "Depression": 0.3, "Stress": 0.5, "Normal": 0.2, "Loneliness": 0.3}
        },
        "Do you have close friends you can confide in?": {
            "recommendation": "Reach out to friends or family members for support.",
            "weights": {"Anxiety": 0.5, "Depression": 0.4, "Stress": 0.3, "Normal": 0.2, "Loneliness": 0.5}
        },
        "Do you spend excessive time on social media?": {
            "recommendation": "Limit your social media usage and engage in offline activities.",
            "weights": {"Anxiety": 0.3, "Depression": 0.2, "Stress": 0.4, "Normal": 0.1, "Loneliness": 0.3}
        },
        "Have you experienced significant weight gain or loss?": {
            "recommendation": "Consult a healthcare professional for a personalized plan.",
            "weights": {"Anxiety": 0.6, "Depression": 0.5, "Stress": 0.7, "Normal": 0.3, "Loneliness": 0.4}
        },
        "Do you place a high value on material possessions?": {
            "recommendation": "Practice gratitude and focus on experiences rather than possessions.",
            "weights": {"Anxiety": 0.1, "Depression": 0.1, "Stress": 0.2, "Normal": 0.1, "Loneliness": 0.1}
        },
        "Do you tend to keep to yourself or prefer solitude?": {
            "recommendation": "Find a balance between alone time and social interactions.",
            "weights": {"Anxiety": 0.3, "Depression": 0.2, "Stress": 0.4, "Normal": 0.1, "Loneliness": 0.3}
        },
        "Do you frequently experience distressing memories?": {
            "recommendation": "Consider therapy or counseling to process traumatic experiences.",
            "weights": {"Anxiety": 0.6, "Depression": 0.5, "Stress": 0.7, "Normal": 0.3, "Loneliness": 0.4}
        },
        "How frequently do you have nightmares?": {
            "recommendation": "Practice relaxation techniques before bedtime to improve sleep quality.",
            "weights": {"Anxiety": 0.4, "Depression": 0.3, "Stress": 0.5, "Normal": 0.2, "Loneliness": 0.3}
        },
        "Do you tend to avoid people or activities?": {
            "recommendation": "Challenge yourself to gradually face situations that cause avoidance.",
            "weights": {"Anxiety": 0.3, "Depression": 0.2, "Stress": 0.4, "Normal": 0.1, "Loneliness": 0.3}
        },
        "How frequently feel negative about yourself or your life?": {
            "recommendation": "Practice self-compassion and challenge negative thoughts.",
            "weights": {"Anxiety": 0.5, "Depression": 0.6, "Stress": 0.4, "Normal": 0.2, "Loneliness": 0.3}
        },
        "Do you have trouble concentrating or focusing?": {
            "recommendation": "Practice mindfulness meditation to improve focus and attention.",
            "weights": {"Anxiety": 0.4, "Depression": 0.3, "Stress": 0.5, "Normal": 0.2, "Loneliness": 0.3}
        },
        "How frequently blame yourself for things?": {
            "recommendation": "Practice self-forgiveness and focus on solutions rather than blame.",
            "weights": {"Anxiety": 0.4, "Depression": 0.3, "Stress": 0.5, "Normal": 0.2, "Loneliness": 0.3}
        },
    }
    return recommendations.get(question, {"recommendation": "No recommendation available", "weights": {"Anxiety": 0, "Depression": 0, "Stress": 0, "Normal": 0, "Loneliness": 0}})


recommendations = {question: get_recommendation_for_question(question) for question in questions}

@app.route('/predict', methods=['POST'])
def predict():
    global client_model
    global recommendations
    global recommendations

    if client_model is None:
        initialize_client_model()

    answers = [request.form.get(f'answer-{i}') for i in range(24)]

    encoded_answers = []
    for ans in answers:
        if ans in ['always', 'often']:
            encoded_answers.append(1)
        else:
            encoded_answers.append(0)

    encoded_answers = []
    for ans in answers:
        if ans in ['always', 'often']:
            encoded_answers.append(1)
        else:
            encoded_answers.append(0)

    input_data = np.array([encoded_answers])

    

    
    predictions = client_model.predict(input_data)

    disorder_labels = ['Anxiety', 'Depression', 'Stress', 'Normal', 'Loneliness']
    predicted_label = np.argmax(predictions)
    prediction_result = disorder_labels[predicted_label]
    
    mapped_recommendations = {}
    for i, answer in enumerate(answers):
        if answer in ['always', 'often']:
            recommendation = questions[i]
            recommendation_data = recommendations[recommendation]  # Retrieve recommendation data from the dictionary
            weight = recommendation_data['weights'][prediction_result]
            mapped_recommendations[recommendation] = {"recommendation": recommendation_data["recommendation"], "weight": weight}

    # Sort mapped recommendations based on their weights
    sorted_recommendations = {k: v for k, v in sorted(mapped_recommendations.items(), key=lambda item: item[1]['weight'], reverse=True)}
    
    # Check if the number of "yes" responses is greater than 8
    num_yes_responses = sum(encoded_answers)
    if num_yes_responses > 8:
        # Display only the top 8 recommendations
        sorted_recommendations = dict(list(sorted_recommendations.items())[:8])
    
    # Pass questions, recommendations, and weights to the HTML template
    return render_template('index.html', questions=questions, prediction=prediction_result, answers=answers, recommendations=sorted_recommendations)


def custom_enumerate(iterable):
    return zip(range(len(iterable)), iterable)

app.jinja_env.globals.update(custom_enumerate=custom_enumerate)

@app.route('/send-email', methods=['POST'])
def send_email():
    data = request.json

    name = data.get('name')
    email = data.get('email')
    message = data.get('message')
    print(f"Name: {name} Email: {email} Message: {message}")

    if not name or not email or not message:
        return jsonify({'error': 'Missing required fields'}), 400
    
    msg = Message(subject='New message from your website',
                  sender=f'mkvaswani90@yahoo.com',
                  recipients=['mkvaswani90@yahoo.com'],
                  body=f"Name: {name}\nEmail: {email}\nMessage: {message}")
    print(f'{msg}')
    try:
        mail.send(msg)
        return jsonify({'message': 'Email sent successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 201


import numpy as np

def update_client_model(input_data, new_target_label):
    global client_model
    if client_model is None:
        # Re-initialize the client model
        initialize_client_model()
    print("INITIAL DATA:", input_data, new_target_label)

   
    # Create output_array
    output_array = np.zeros(5)
    output_array[new_target_label] = 1

    concatenated_data = input_data + output_array.tolist()
    print("CONCATENATED DATA :",concatenated_data)
    data = {f'col_{i}': [val] for i, val in enumerate(concatenated_data)}
    df = pd.DataFrame(data)


    X = df.iloc[:, :-5]  # Select all columns except the last 5
    Y = df.iloc[:, -5:]   # Select only the last 5 columns


    
    print("NEW DATA SHAPE:", X.shape, Y.shape)
    print("MODEL SUMMARY:", client_model.summary())
    print("INITIAL DATA:", input_data, new_target_label)

   
    # Create output_array
    output_array = np.zeros(5)
    output_array[new_target_label] = 1

    concatenated_data = input_data + output_array.tolist()
    print("CONCATENATED DATA :",concatenated_data)
    data = {f'col_{i}': [val] for i, val in enumerate(concatenated_data)}
    df = pd.DataFrame(data)


    X = df.iloc[:, :-5]  # Select all columns except the last 5
    Y = df.iloc[:, -5:]   # Select only the last 5 columns


    
    print("NEW DATA SHAPE:", X.shape, Y.shape)
    print("MODEL SUMMARY:", client_model.summary())
    # Compile the model if it hasn't been compiled already
    if not client_model._jit_compile:
    if not client_model._jit_compile:
        client_model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Train the client model on the new data
    client_model.fit(X, Y, epochs=1, batch_size=1, verbose=0)
    print("DONEEEEEEEE")
    send_updated_model_to_server()


def send_updated_model_to_server():
    global client_model

    # Extract the updated model weights
    updated_weights = client_model.get_weights()
    client_model.save_weights('client_model_weights.weights.h5')

    # Send the updated model weights file to the server for aggregation
    files = {'file': open('client_model_weights.h5', 'rb')}
    response = requests.post('http://localhost:5000/update_model', files=files)

    return 'Updated model weights sent to server for aggregation!'

@app.route('/generate-ai-answer', methods=['POST'])
def genResp():
    
    question = request.json.get('question')
    print("QUES:",question)
    question += "precisely in 2 or 3 lines in context of mental health. If my context is beyod mental health then say I didn't get your question"
    if question:
        response = model.generate_content(question)
        candidates = response.candidates 
        if candidates:  
            content = candidates[0].content  
            parts = content.parts  
            if parts: 
                text_answer = parts[0].text
                return text_answer

    return "No text answer found"


@app.route('/finish')
def finish():
    # Render the finish template
    return render_template('finish.html')

if __name__ == '__main__':  
    app.run(debug=True, port=5001)