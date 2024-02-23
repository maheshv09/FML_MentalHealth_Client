import React, { useState, useEffect } from "react";
import { View, Text, Button } from "react-native";
import axios from "axios";

const App = () => {
  const [questionNumber, setQuestionNumber] = useState(1);
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    // Fetch basic ML model from backend
    axios
      .get("http://127.0.0.1:5000//get_model")
      .then((response) => {
        console.log("MKK", response);
        setModel(response.data);
        // Set up initial question(s) based on model or hardcoded
        setQuestions([
          "Do you often feel nervous?",
          "Do you experience panic attacks?",
          // Add more questions based on your dataset
        ]);
      })
      .catch((error) => {
        console.error("Error fetching basic model:", error);
      });
  }, []);

  const handleAnswer = (question, answer) => {
    // Update local answers state
    setAnswers((prevAnswers) => ({ ...prevAnswers, [question]: answer }));

    // Move to the next question or submit answers if all questions answered
    if (questionNumber < questions.length) {
      setQuestionNumber(questionNumber + 1);
    } else {
      // Make prediction using local model
      const localPrediction = predictWithLocalModel(model, answers);
      setPrediction(localPrediction);

      // Send model updates to backend
      axios
        .post("http://your-backend-url/update_model", { model_update: model })
        .then((response) => {
          console.log("Model updates sent successfully");
        })
        .catch((error) => {
          console.error("Error sending model updates:", error);
        });
    }
  };

  // Function to make prediction using local model
  const predictWithLocalModel = (model, answers) => {
    if (!model) {
      console.error("No model available for prediction");
      return null;
    }

    // Convert answers to the format expected by the model (if needed)
    // Assuming RandomForestClassifier expects a 2D array where each row represents a sample and each column represents a feature
    const formattedAnswers = Object.values(answers).map((answer) => [answer]);

    // Make prediction using the fetched model
    const prediction = model.predict(formattedAnswers);

    return prediction; // Return the prediction
  };

  return (
    <View>
      {questions.length > 0 && questionNumber <= questions.length ? (
        <View>
          <Text>{questions[questionNumber - 1]}</Text>
          <Button
            title="Yes"
            onPress={() => handleAnswer(questions[questionNumber - 1], true)}
          />
          <Button
            title="No"
            onPress={() => handleAnswer(questions[questionNumber - 1], false)}
          />
        </View>
      ) : (
        <View>
          <Text>Thank you for answering the questions.</Text>
          {prediction && <Text>Prediction: {prediction}</Text>}
        </View>
      )}
    </View>
  );
};

export default App;
