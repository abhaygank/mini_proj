from keras.models import model_from_json
import cv2
import numpy as np

# Load the first model (Gesture recognition)
json_file_1 = open("w1signlanguagejustword.json", "r")
model_json_1 = json_file_1.read()
json_file_1.close()
model_1 = model_from_json(model_json_1)
model_1.load_weights("w1signlanguagejustword.h5")

# Load the second model (Alphabet recognition)
json_file_2 = open("w1signlanguagejustalpha.json", "r")
model_json_2 = json_file_2.read()
json_file_2.close()
model_2 = model_from_json(model_json_2)
model_2.load_weights("w1signlanguagejustalpha.h5")

# Define label sets for each model
labels_model_1 = ['HELLO', 'I LOVE YOU', 'NO', 'WHICH', 'YES']
labels_model_2 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
# Function to preprocess the image for the models
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start capturing video
cap = cv2.VideoCapture(0)

# Variable to control which model to use
use_model_1 = True  # Start with the first model

while True:
    _, frame = cap.read()
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    crop_frame = frame[40:300, 0:300]
    crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    crop_frame = cv2.resize(crop_frame, (48, 48))
    processed_frame = extract_features(crop_frame)
    
    if use_model_1:
        # Predict using the first model (Gesture recognition)
        pred_1 = model_1.predict(processed_frame)
        prediction_label_1 = labels_model_1[pred_1.argmax()]
        confidence_1 = "{:.2f}".format(np.max(pred_1) * 100)

        # Display result for model 1
        text_1 = f'M1: {prediction_label_1} ({confidence_1}%)'
        cv2.putText(frame, text_1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # Predict using the second model (Alphabet recognition)
        pred_2 = model_2.predict(processed_frame)
        prediction_label_2 = labels_model_2[pred_2.argmax()]
        confidence_2 = "{:.2f}".format(np.max(pred_2) * 100)

        # Display result for model 2
        text_2 = f'M2: {prediction_label_2} ({confidence_2}%)'
        cv2.putText(frame, text_2, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Toggle model switch after a short interval or when a condition is met
    # Example: Every 5 seconds or based on a gesture
    if cv2.waitKey(1) & 0xFF == ord('m'):  # Press 'm' to manually switch models
        use_model_1 = not use_model_1

    # Show the frame with prediction
    cv2.imshow("output", frame)

    key = cv2.waitKey(27)
    if key == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
