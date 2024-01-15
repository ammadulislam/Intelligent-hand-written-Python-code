from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage

app = Flask(__name__)

class CharacterRecognition:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def preprocess_image(self, char_image):
        # Resize the image to 28x28
        img = PILImage.fromarray(char_image)
        img = img.resize((28, 28))

        # Create a new image with a black background
        new_img = PILImage.new('L', img.size, 'black')

        # Paste the character (white) onto the black background
        new_img.paste(img, (0, 0))

        img = np.array(new_img)
        img = np.expand_dims(img, axis=0)  # Expand along the first axis
        img = img / 255.0
        return img

    def process_image(self, img):
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to segment characters
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Use morphological operations to remove small noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set the text regions to white
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = binary[y:y+h, x:x+w]
            img[y:y+h, x:x+w][roi == 255] = [255, 255, 255]  # White

        # Set the background to black
        img[binary == 0] = [0, 0, 0]

        # Now, process the image to extract characters with bounding boxes
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Adjust the area threshold as needed
                (x, y, w, h) = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))

        # Calculate the average bounding box size
        average_box_size = np.mean([w * h for (x, y, w, h) in bounding_boxes])

        # Set your scaling factor (you may need to adjust this based on your specific use case)
        scaling_factor = 0.1

        # Calculate the font size
        font_size = scaling_factor * average_box_size

        bounding_boxes.sort(key=lambda x: x[0])

        predicted_characters = []
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            char_image = gray_image[y:y + h, x:x + w]
            processed_image = self.preprocess_image(char_image)
            prediction = self.model.predict(processed_image)
            predicted_class_index = np.argmax(prediction)

            # Calculate font size based on individual bounding box size
            char_font_size = scaling_factor * w * h

            # Append the predicted character and font size to the list as a dictionary
            predicted_characters.append({'character': int(predicted_class_index), 'font_size': float(char_font_size)})

        return predicted_characters
# Define the character_recognizer outside the Flask app block
character_recognizer = CharacterRecognition(
    model_path='C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Work\\mnist_model.h5')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return {"error": "No image provided"}

        image_file = request.files['image']
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Assuming character_recognizer is an instance of your CharacterRecognition class
        predicted_characters = character_recognizer.process_image(img)

        # Extract 'character' values and convert them to integers
        predicted_characters = [{'character': int(char['character']), 'font_size': char['font_size']} for char in
                                predicted_characters]

        return jsonify({'predicted_characters': predicted_characters})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
