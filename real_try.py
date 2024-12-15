import cv2
import numpy as np
from keras.models import model_from_json

# Load the model
json_file = open("w1signlanguagedetectionmodel48x48.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("w1signlanguagedetectionmodel48x48.h5")

# Initialize text-to-speech engine


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Define the labels for detection
label =['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'hello', 'iloveyou', 'no', 'nothing', 'space', 'which', 'yes']


# Set up the webcam capture
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    # Draw rectangle around the detected sign
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    
    # Crop, convert to grayscale, and resize the image
    cropframe = frame[40:300, 0:300]
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    cropframe = cv2.resize(cropframe, (48, 48))
    
    # Extract features and predict the sign
    cropframe = extract_features(cropframe)
    pred = model.predict(cropframe)
    prediction_label = label[pred.argmax()]
    
    # Display the result on the frame
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    
    # Display the detected sign and confidence
    if prediction_label == 'nothing':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred) * 100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
       
    
    # Show the output frame
    cv2.imshow("output", frame)
    
    # Break the loop when 'Esc' key is pressed
    if cv2.waitKey(27) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
