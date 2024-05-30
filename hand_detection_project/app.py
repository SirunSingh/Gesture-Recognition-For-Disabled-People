from flask import Flask, render_template, jsonify
import subprocess
import threading

app = Flask(__name__)

# To store the result of the gesture recognition
recognition_result = ""

def run_gesture_recognition():
    global recognition_result
    try:
        result = subprocess.run(['python', 'C:/Users/Anubhav/Desktop/hand_detection_project/recognition.py'], capture_output=True, text=True)
        recognition_result = result.stdout
    except Exception as e:
        recognition_result = f"Error: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize_gesture', methods=['POST'])
def recognize_gesture():
    global recognition_result
    # Reset the result
    recognition_result = ""
    # Start the gesture recognition in a new thread
    thread = threading.Thread(target=run_gesture_recognition)
    thread.start()
    # Return a message to indicate the process has started
    return jsonify(message="Gesture recognition started. Please wait for the result.")

@app.route('/get_result', methods=['GET'])
def get_result():
    return jsonify(result=recognition_result)

if __name__ == '__main__':
    app.run(debug=True)
