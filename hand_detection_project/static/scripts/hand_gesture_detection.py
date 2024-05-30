import cv2

# Segment the hand region in the image
def segment(image, threshold=25):
    global bg
    
    # Compute the absolute difference between the background and the current image
    diff = cv2.absdiff(bg.astype("uint8"), image)
    
    # Apply a binary threshold to get the foreground
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours are found, return None
    if not contours:
        return None
    
    # Find the largest contour by area, assumed to be the hand
    largest_contour = max(contours, key=cv2.contourArea)
    
    return thresholded, largest_contour

import cv2
import numpy as np
from keras.models import load_model

# Initialize background as None
bg = None

# Function to update the running average of the background
def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)

# Load the pre-trained model
def _load_weights():
    try:
        model = load_model("model6.h5")
        print(model.summary())
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Predict the gesture class
def getPredictedClass(model):
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (100, 120))
    gray_image = gray_image.reshape(1, 100, 120, 1)

    prediction = model.predict_on_batch(gray_image)
    predicted_class = np.argmax(prediction)
    
    classes = ["Blank", "OK", "Thumbs Up", "Thumbs Down", "Punch", "High Five"]
    return classes[predicted_class]

# Function to segment the hand region
def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return thresholded, largest_contour

if __name__ == "__main__":
    # Initialize accumulated weight
    accumWeight = 0.5

    # Get the reference to the webcam
    camera = cv2.VideoCapture(0)

    fps = int(camera.get(cv2.CAP_PROP_FPS))
    # Region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590
    # Initialize number of frames
    num_frames = 0
    # Calibration indicator
    calibrated = False
    model = _load_weights()
    k = 0
    
    # Variables to hold the last stable predicted class
    stable_class = ""
    stable_counter = 0
    stable_threshold = 10  # Number of frames to confirm stability
    display_class = ""  # To store the class to be displayed

    # Keep looping, until interrupted
    while True:
        # Get the current frame
        grabbed, frame = camera.read()

        if not grabbed:
            print("Error: Could not read frame.")
            break

        # Resize the frame
        frame = cv2.resize(frame, (700, 700))
        # Flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # Clone the frame
        clone = frame.copy()

        # Get the ROI
        roi = frame[top:bottom, right:left]

        # Convert the ROI to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # To get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successful...")
        else:
            # Segment the hand region
            hand = segment(gray)

            # Check whether hand region is segmented
            if hand is not None:
                # If yes, unpack the thresholded image and
                # segmented region
                thresholded, segmented = hand

                # Draw the segmented region and display the frame
                display_contour = segmented + (right, top)
                cv2.drawContours(clone, [display_contour], -1, (0, 0, 255))

                if k % (fps // 6) == 0:
                    cv2.imwrite('Temp.png', thresholded)
                    predicted_class = getPredictedClass(model)

                    # Check for stability
                    if predicted_class == stable_class:
                        stable_counter += 1
                    else:
                        stable_counter = 0
                        stable_class = predicted_class

                    if stable_counter >= stable_threshold:
                        display_class = stable_class

                # Show the thresholded image
                cv2.imshow("Thresholded", thresholded)

        k += 1
        # Draw the ROI rectangle
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the stable class
        cv2.putText(clone, str(display_class), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Increment the number of frames
        num_frames += 1

        # Display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # Observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # If the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

    # Free up memory
    camera.release()
    cv2.destroyAllWindows()
