import pyodbc
from flask import Flask, request, jsonify,send_file
from flask import Flask, jsonify, send_from_directory
app = Flask(__name__)
server = 'DESKTOP-TJRF2CO'
database = 'FYPDatabase'
username = 'sa'
password = '123'

def connect_to_database():
    connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    return pyodbc.connect(connection_string)
try:
    # Attempt to establish a connection
    connection = connect_to_database()
    print("Connection to SQL Server successful")
    connection.close()
except Exception as e:
    print(f"Error: {str(e)}")



@app.route("/pdf/<string:filename>")
def filePath(filename):
    return send_file(filename)


@app.route('/getPDFs', methods=['GET'])
def get_pdfs():
    try:
        connection = connect_to_database()
        cursor = connection.cursor()

        # Retrieve all PDF paths from the Document table
        cursor.execute("SELECT DocumentID, DocumentPath,Title FROM Document WHERE DocumentPath IS NOT NULL")
        pdf_records = cursor.fetchall()

        connection.close()

        # Create a list of dictionaries containing document IDs and paths
        pdf_list = [{"document_id": record[0], "pdf_path": record[1],"Title":record[2]} for record in pdf_records]

        return jsonify({"pdfs": pdf_list})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An error occurred while processing your request"}), 500









# # Assuming the base directory where PDF folders are stored
# base_directory = 'C:/Users/HP/PycharmProjects/Fyp_Final/Data'
#
# @app.route('/getPDFs', methods=['GET'])
# def get_pdfs():
#     try:
#         connection = connect_to_database()
#         cursor = connection.cursor()
#
#         # Retrieve all PDF paths and IDs from the Document table
#         cursor.execute("SELECT DocumentID, DocumentPath, Title FROM Document WHERE DocumentPath IS NOT NULL")
#         pdf_records = cursor.fetchall()
#
#         connection.close()
#
#         # Create a list of dictionaries containing document IDs, project IDs, and titles
#         pdf_list = [{"document_id": record[0], "DocumentPath": record[1], "title": record[2]} for record in pdf_records]
#
#         return jsonify({"pdfs": pdf_list})
#
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": "An error occurred while processing your request"}), 500
#
# @app.route("/pdf/<int:document_id>")
# def serve_pdf(document_id):
#     try:
#         connection = connect_to_database()
#         cursor = connection.cursor()
#
#         # Retrieve the project ID based on the document ID
#         cursor.execute("SELECT DocumentID FROM Document WHERE DocumentID = %s", (document_id,))
#         project_id = cursor.fetchone()[0]
#
#         connection.close()
#
#         # Construct the PDF path based on the project ID and document ID
#         pdf_path = f"{base_directory}/{project_id}/ProjectData/{document_id}.pdf"
#
#         return send_from_directory(base_directory, pdf_path, as_attachment=True)
#
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": "An error occurred while serving the PDF"}), 500
#
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=False)












































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
#         char_data = []  # List to store character information
#         for i, (x, y, w, h) in enumerate(bounding_boxes):
#             # Crop the character from the grayscale image
#             char_image = gray_image[y:y + h, x:x + w]
#
#             # Preprocess the character image and make a prediction using the model
#             processed_image = self.preprocess_image(char_image)
#             prediction = self.model.predict(processed_image)
#             predicted_class_index = np.argmax(prediction)
#             # Append character information to the list
#             char_data.append((predicted_class_index, x_coordinate, y_coordinate, color_code))
#
#             # Trigger PDF generation with predicted characters in a single file
#         pdf_file_path = os.path.join(pdf_directory, "output_combined.pdf")
#         generate_pdf_thread = Thread(target=generate_pdf, args=(
#             pdf_file_path, custom_width, custom_height, char_data))
#         generate_pdf_thread.start()
#
#         print(f"PDF saved to: {pdf_file_path}")
#
#         return jsonify({'predicted_characters': predicted_characters})
#
#
# character_recognizer = CharacterRecognition(model_path='C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Work\\mnist_model.h5')
#
# predicted_characters = []  # Replace with your actual list
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
#         for char in predicted_characters:
#             # Get user input for x and y coordinates for each character
#             x_coordinate = float(input(f"Enter X coordinate for {char}: "))
#             y_coordinate = float(input(f"Enter Y coordinate for {char}: "))
#             color_code = input(f"Enter color code for {char} (e.g., ff2ee615): ")
#
#             # Set the PDF file path
#             pdf_file_path = os.path.join(pdf_directory, f"output_{char}.pdf")
#
#             # Trigger PDF generation with predicted characters
#             generate_pdf_thread = Thread(target=generate_pdf, args=(
#                 pdf_file_path, custom_width, custom_height, char, x_coordinate, y_coordinate, color_code))
#             generate_pdf_thread.start()
#
#             print(f"PDFs saved to: {pdf_directory}")
#
#         return jsonify({'predicted_characters': predicted_characters})
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
#
# def generate_pdf(pdf_file_path, width, height, char_data):
#     c = canvas.Canvas(pdf_file_path, pagesize=(width, height))
#
#     for char_info in char_data:
#         char, x, y, color_code = char_info
#
#         # Check if the color code starts with '0x' and remove it if present
#         if color_code.startswith('0x'):
#             color_code = color_code[2:]
#
#         # Extract RGB values from the color code
#         rgb_color = [(int(color_code[i:i + 2], 16) / 255.0) for i in (0, 2, 4)]
#
#         # Set fill color
#         c.setFillColorRGB(*rgb_color)
#
#         # Calculate the width and height of the text
#         text_object = c.beginText(0, 0)
#         text_object.setFont("Helvetica", 12)  # You can adjust the font and size
#         text_object.textLine(f"{char}")
#         text_width = c.stringWidth(f"{char}", "Helvetica", 12)
#         text_height = 12  # Assuming font size is 12
#
#         # Add user-entered text to the PDF at the specified x, y coordinates
#         c.drawString(x - text_width / 2, height - y - text_height / 2, f"{char}")
#
#     try:
#         print(f"Attempting to save PDF to: {pdf_file_path}")
#         c.save()
#         print("PDF saved successfully.")
#     except Exception as e:
#         print(f"Error saving PDF: {e}")
# def run_flask_app():
#     app.run(debug=True, use_reloader=False)
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
#         # Assuming predicted_characters is a list of characters
#         predicted_characters = []  # Replace with your actual list
#
#         char_data = []  # List to store character information
#
#         for char in predicted_characters:
#             # Get user input for x and y coordinates for each character
#             x_coordinate = float(input(f"Enter X coordinate for {char}: "))
#             y_coordinate = float(input(f"Enter Y coordinate for {char}: "))
#             color_code = input(f"Enter color code for {char} (e.g., ff2ee615): ")
#
#             # Validate input to ensure it is not an empty string
#             if not x_coordinate or not y_coordinate or not color_code:
#                 print("Invalid input. Please enter valid values.")
#                 continue
#
#             # Append character information to the list
#             char_data.append((char, x_coordinate, y_coordinate, color_code))
#
#         # Trigger PDF generation with predicted characters in a single file
#         pdf_file_path = os.path.join(pdf_directory, "output_combined.pdf")
#         generate_pdf_thread = Thread(target=generate_pdf, args=(
#             pdf_file_path, custom_width, custom_height, char_data))
#         generate_pdf_thread.start()
#
#         print(f"PDF saved to: {pdf_file_path}")
#
#     except KeyboardInterrupt:
#         # Stop the Flask app when KeyboardInterrupt (Ctrl+C) is detected
#         flask_thread.join()
#
#     except KeyboardInterrupt:
#         # Stop the Flask app when KeyboardInterrupt (Ctrl+C) is detected
#         flask_thread.join()
#
#
#         def save_to_pdf(self, output_pdf_path, char_images_with_coordinates, input_image_path, slide_num, slide_count):
#             img_width, img_height = self.get_image_dimensions(input_image_path)
#
#             colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Add more colors as needed
#             for ((char_output_filename, (x, y, w, h)), predicted_char_index) in char_images_with_coordinates:
#
#                 # Determine font size based on character height
#                 if 110 <= h < 120:
#                     font_size = 60
#                 elif 120 <= h < 130:
#                     font_size = 62
#                 elif 130 <= h < 140:
#                     font_size = 68
#                 elif 140 <= h < 150:
#                     font_size = 72
#                 elif 150 <= h < 160:
#                     font_size = 78
#                 elif 160 <= h < 170:
#                     font_size = 80
#                 else:
#                     # Default font size if height is out of specified ranges
#                     font_size = 16
#
#                 # Get the color based on the predicted character index
#                 color = colors[predicted_char_index % len(colors)]
#
#                 # Check if the current slide number is less than or equal to the total slide count
#                 if slide_num <= slide_count:
#                     # Create a new PDF instance for each slide
#                     pdf = canvas.Canvas(output_pdf_path, pagesize=(img_width, img_height))
#                     # Draw the character label on the PDF at the specified coordinates with the determined font size and color
#                     pdf.setFillColorRGB(*color)
#                     pdf.setFont("Helvetica", font_size)
#                     pdf.drawString(x, img_height - y - h - 10, f"{predicted_char_index}")
#
#                     # Check if the current slide number is equal to the total slide count
#                     if slide_num == slide_count:
#                         # Save the PDF when all data for the current slide has been processed
#                         pdf.save()
#
#                     # Increment the slide number for the next iteration
#                     slide_num += 1
#
#                     # Create a new page for the next slide if there are more slides to process
#                     if slide_num <= slide_count:
#                         pdf.showPage()
#
#                         def save_to_pdf(self, output_pdf_path, char_images_with_coordinates, input_image_path,
#                                         slide_num, slide_count):
#                             print("first")
#                             print(slide_num)
#                             img_width, img_height = self.get_image_dimensions(input_image_path)
#
#                             pdf = canvas.Canvas(output_pdf_path, pagesize=(img_width, img_height))
#
#                             colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Add more colors as needed
#
#                             for (
#                             (char_output_filename, (x, y, w, h)), predicted_char_index) in char_images_with_coordinates:
#                                 # Determine font size based on character height
#                                 if 110 <= h < 120:
#                                     font_size = 60
#                                 elif 120 <= h < 130:
#                                     font_size = 62
#                                 elif 130 <= h < 140:
#                                     font_size = 68
#                                 elif 140 <= h < 150:
#                                     font_size = 72
#                                 elif 150 <= h < 160:
#                                     font_size = 78
#                                 elif 160 <= h < 170:
#                                     font_size = 80
#                                 else:
#                                     # Default font size if height is out of specified ranges
#                                     font_size = 16
#
#                                 # Draw the character label on the PDF at the specified coordinates with the determined font size and color
#                                 color = colors[predicted_char_index % len(colors)]
#                                 pdf.setFillColorRGB(*color)
#                                 pdf.setFont("Helvetica", font_size)
#                                 pdf.drawString(x, img_height - y - h - 10, f"{predicted_char_index}")
#
#                                 # Create a new page for the next slide if there are more slides to process
#                             if slide_num <= slide_count:
#                                 pdf.showPage()
#                                 print("new page ")
#                                 # Save the PDF after processing all slides
#                             if slide_num == slide_count:
#                                 pdf.save()
#                                 print("Pdf end")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)