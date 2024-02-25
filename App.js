import React, { useState, useEffect } from "react";
import { View, Text, Button } from "react-native";
//import { RandomForestClassifier } from "scikit-learn";
import axios from "axios";

const App = () => {
  const [questionNumber, setQuestionNumber] = useState(1);
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [modelParam, setModelParam] = useState(null);

  useEffect(() => {
    //Fetch basic ML model from backend
    axios
      .get("http://localhost:5000/get_model")
      .then((response) => {
        console.log("MKK", response);
        setModelParam(response);
        //setModel(new RandomForestClassifier(modelParam));
        // Set up initial question(s) based on model or hardcoded
        setQuestions([
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
          "Do you often blame yourself for things?",
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
      // axios
      //   .post("https://fml-mentalhealth-backend-1.onrender.com/update_model", {
      //     model_update: model,
      //   })
      //   .then((response) => {
      //     console.log("Model updates sent successfully");
      //   })
      //   .catch((error) => {
      //     console.error("Error sending model updates:", error);
      //   });
    }
  };

  // Function to make prediction using local model
  const predictWithLocalModel = async (model, answers) => {
    if (!model) {
      console.error("No model available for prediction");
      return null;
    }

    // Convert answers to the format expected by the model (if needed)
    // Assuming RandomForestClassifier expects a 2D array where each row represents a sample and each column represents a feature
    const formattedAnswers = Object.values(answers).map((answer) => [answer]);
    console.log("Answers", formattedAnswers);
    // Make prediction using the fetched model
    const prediction = await model.predict(formattedAnswers);
    console.log("Prediction", prediction);
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
      {/* <Text>Hello ghchg</Text>
      <Text>Hello ghchg</Text>
      <Text>Hello ghchg</Text>
      <Text>Hello ghchg</Text>
      <Text>Hello ghchg</Text>
      <Text>Hello ghchg</Text>
      <Text>Hello ghchg</Text>
      <Text>Hello ghchg</Text>
      <Text>Hello ghchg</Text>
      <Text>Hello ghchg</Text> */}
    </View>
  );
};

export default App;
