# import os
# import time
#
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# import tensorflow as tf
# from PIL import Image as PILImage
# from io import BytesIO
# from threading import Thread
# import random
#
# app = Flask(__name__)
#
# class CharacterRecognition:
#     def __init__(self, model_path):
#         self.model = tf.keras.models.load_model(model_path)
#
#     def preprocess_image(self, char_image):
#         # Resize the image to 28x28
#         img = PILImage.fromarray(char_image)
#         img = img.resize((28, 28))
#
#         # Create a new image with a black background
#         new_img = PILImage.new('L', img.size, 'black')
#
#         # Paste the character (white) onto the black background
#         new_img.paste(img, (0, 0))
#
#         img = np.array(new_img)
#         img = np.expand_dims(img, axis=0)  # Expand along the first axis
#         img = img / 255.0
#         return img
#
#     def process_image(self, img):
#         # Convert the image to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # Apply adaptive thresholding to segment characters
#         binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#
#         # Use morphological operations to remove small noise
#         kernel = np.ones((3, 3), np.uint8)
#         binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
#
#         # Find contours in the binary image
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         # Set the text regions to white
#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             roi = binary[y:y + h, x:x + w]
#             img[y:y + h, x:x + w][roi == 255] = [255, 255, 255]  # White
#
#         # Set the background to black
#         img[binary == 0] = [0, 0, 0]
#
#         # Now, process the image to extract characters with bounding boxes
#         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,
#                                              2)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#         cleaned_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
#         contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         bounding_boxes = []
#
#         for contour in contours:
#             if cv2.contourArea(contour) > 50:  # Adjust the area threshold as needed
#                 (x, y, w, h) = cv2.boundingRect(contour)
#                 bounding_boxes.append((x, y, w, h))
#
#         bounding_boxes.sort(key=lambda x: x[0])
#
#         predicted_characters = []
#
#         for i, (x, y, w, h) in enumerate(bounding_boxes):
#             # Crop the character from the grayscale image
#             char_image = gray_image[y:y + h, x:x + w]
#
#             # Preprocess the character image and make a prediction using the model
#             processed_image = self.preprocess_image(char_image)
#             prediction = self.model.predict(processed_image)
#             predicted_class_index = np.argmax(prediction)
#
#             # Append the predicted character to the list
#             predicted_characters.append(predicted_class_index)
#
#         return predicted_characters
#
# character_recognizer = CharacterRecognition(model_path='C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Work\\mnist_model.h5')
#
# predicted_characters = []  # Replace with your actual list
# character_info_list = []
#
# @app.route('/process_image', methods=['POST'])
# def process_image():
#     try:
#         if 'image' not in request.files:
#             return {"error": "No image provided"}
#
#         image_file = request.files['image']
#         img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
#
#         predicted_characters = character_recognizer.process_image(img)
#         predicted_characters = [int(x) for x in predicted_characters]
#
#         for i, char in enumerate(predicted_characters):
#             # Get user input for x and y coordinates for each character
#             x_coordinate = random.uniform(0, custom_width)
#             y_coordinate = random.uniform(0, custom_height)
#
#             # Generate a random color code
#             color_code = random.randint(0, 0xFFFFFF)
#
#             # Convert color code to hexadecimal format
#             color_code = f"0x{color_code:06X}"
#
#             # Add character information to the list
#             char_info = {
#                 'char': char,
#                 'x_coordinate': x_coordinate,
#                 'y_coordinate': y_coordinate,
#                 'color_code': color_code,
#                 'slide_num': 1  # Use slide/page numbers starting from 1
#             }
#
#             character_info_list.append(char_info)
#
#             # Set the PDF file path
#             pdf_file_path = os.path.join(pdf_directory, f"output.pdf")
#
#             # Trigger PDF generation with predicted characters
#             generate_pdf_thread = Thread(target=generate_pdf, args=(
#                 pdf_file_path, custom_width, custom_height, char_info))
#             generate_pdf_thread.start()
#
#             print(f"PDFs saved to: {pdf_directory}")
#
#     except Exception as e:
#         print(f"Error processing image: {e}")
#
# def generate_pdf(pdf_file_path, width, height, char_info):
#     char = char_info['char']
#     x = char_info['x_coordinate']
#     y = char_info['y_coordinate']
#     color_code = char_info['color_code']
#
#     c = canvas.Canvas(pdf_file_path, pagesize=(width, height))
#
#     # Check if the color code starts with '0x' and remove it if present
#     if color_code.startswith('0x'):
#         color_code = color_code[2:]
#
#     # Extract RGB values from the color code
#     rgb_color = [(int(color_code[i:i + 2], 16) / 255.0) for i in (0, 2, 4)]
#
#     # Set fill color
#     c.setFillColorRGB(*rgb_color)
#
#     # Calculate the width and height of the text
#     text_object = c.beginText(0, 0)
#     text_object.setFont("Helvetica", 12)  # You can adjust the font and size
#     text_object.textLine(f"{char}")
#     text_width = c.stringWidth(f"{char}", "Helvetica", 12)
#     text_height = 12  # Assuming font size is 12
#
#     # Add user-entered text to the PDF at the specified x, y coordinates
#     c.drawString(x - text_width / 2, height - y - text_height / 2, f"{char}")
#
#     try:
#         print(f"Attempting to save PDF to: {pdf_file_path}")
#         c.save()
#         print("PDF saved successfully.")
#     except Exception as e:
#         print(f"Error saving PDF: {e}")
#
# def run_flask_app():
#     app.run(debug=True, use_reloader=False)
#
# if __name__ == "__main__":
#     # Set the path for the PDF files (use an absolute path)
#     pdf_directory = r"C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs"
#
#     # Create the directory if it doesn't exist
#     os.makedirs(pdf_directory, exist_ok=True)
#
#     # Custom page size (Width: 342.4285714285, Height: 730.1428571428571)
#     custom_width = 342.4285714285
#     custom_height = 730.1428571428571
#
#     # Start Flask app in a separate thread
#     flask_thread = Thread(target=run_flask_app)
#     flask_thread.start()
#
#     try:
#         while True:
#             if c"C:\Program Files\Python311\python.exe" C:\Users\HP\PycharmProjects\Fyp_Final\main.py
#  * Serving Flask app 'main'
#  * Debug mode: on
# WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
#  * Running on http://127.0.0.1:5000
# Press CTRL+C to quit
# 1/1 [==============================] - 0s 280ms/step
# 1/1 [==============================] - 0s 72ms/step
# 1/1 [==============================] - 0s 53ms/step
# 1/1 [==============================] - 0s 48ms/step
# 1/1 [==============================] - 0s 77ms/step
# 1/1 [==============================] - 0s 32ms/step
# 1/1 [==============================] - 0s 80ms/step
# 1/1 [==============================] - 0s 58ms/step
# 1/1 [==============================] - 0s 33ms/step
# PDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdf
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdf
# PDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdf
# PDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdfPDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfsPDF saved successfully.
# PDF saved successfully.
#
#
# PDF saved successfully.
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdf
# PDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
# PDF saved successfully.
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdf
# PDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdfPDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
#
# PDF saved successfully.
# PDF saved successfully.
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdf
# PDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
# PDF saved successfully.
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdf
# PDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
# PDF saved successfully.
# PDF saved successfully.
# 127.0.0.1 - - [12/Nov/2023 23:05:34] "POST /process_image HTTP/1.1" 500 -
# Traceback (most recent call last):
#   File "C:\Program Files\Python311\Lib\site-packages\flask\app.py", line 1478, in __call__
#     return self.wsgi_app(environ, start_response)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Program Files\Python311\Lib\site-packages\flask\app.py", line 1458, in wsgi_app
#     response = self.handle_exception(e)
#                ^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Program Files\Python311\Lib\site-packages\flask\app.py", line 1455, in wsgi_app
#     response = self.full_dispatch_request()
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Program Files\Python311\Lib\site-packages\flask\app.py", line 870, in full_dispatch_request
#     return self.finalize_request(rv)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Program Files\Python311\Lib\site-packages\flask\app.py", line 889, in finalize_request
#     response = self.make_response(rv)
#                ^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Program Files\Python311\Lib\site-packages\flask\app.py", line 1161, in make_response
#     raise TypeError(
# TypeError: The view function for 'process_image' did not return a valid response. The function either returned None or ended without a return statement.
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdfPDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
#
# PDF saved successfully.
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdfPDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
#
# PDF saved successfully.
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdf
# PDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
# PDF saved successfully.
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdfPDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
#
# PDF saved successfully.
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdfPDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
#
# PDF saved successfully.
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdf
# PDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
# PDF saved successfully.
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdfPDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
#
# PDF saved successfully.
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdfPDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
#
# PDF saved successfully.
# Attempting to save PDF to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs\output.pdfPDFs saved to: C:\Users\HP\PycharmProjects\Fyp_Final\Work\output_pdfs
#
# PDF saved successfully.
# haracter_info_list:
#                 char_info = character_info_list.pop(0)
#                 pdf_file_path = os.path.join(pdf_directory, f"output.pdf")
#
#                 # Trigger PDF generation with predicted characters
#                 generate_pdf_thread = Thread(target=generate_pdf, args=(
#                     pdf_file_path, custom_width, custom_height, char_info))
#                 generate_pdf_thread.start()
#
#                 print(f"PDFs saved to: {pdf_directory}")
#             time.sleep(1)  # Adjust sleep time as needed
#
#     except KeyboardInterrupt:
#         # Stop the Flask app when KeyboardInterrupt (Ctrl+C) is detected
#         flask_thread.join()
