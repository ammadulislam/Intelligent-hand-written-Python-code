import cv2
import os
import numpy as np
import self
import tensorflow as tf
from PIL import Image as PILImage, ImageColor
from flask import Flask, jsonify, request
import datetime
from flask import Flask, jsonify
from flask_cors import CORS
from reportlab.pdfgen import canvas

from Test import StrokeObject
app = Flask(__name__)
CORS(app)
import base64
from PIL import Image
from io import BytesIO
import json
import pyodbc
import re

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


def get_current_time():
    return datetime.datetime.now()

class StrokeObject:
    def __init__(self, x, y, thickness, color):
        self.x = x
        self.y = y
        self.thickness= thickness
        self.color = color
char_images_with_coordinates = []
class CharacterRecognition:


    def __init__(self, model_path):
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print("Error loading the model:", str(e))

        self.image = None  # Initialize the 'image' attribute

        # Initialize the current page attribute
        self.current_page = 1
    def predict_character_index(self, char_image):
        # Resize the character image to match the model input size
        char_image = cv2.resize(char_image, (28, 28))

        # Preprocess the image for the model
        char_image = self.preprocess_image(char_image)

        # Make predictions using the loaded model
        predictions = self.model.predict(char_image)

        # Get the index with the highest probability as the predicted character index
        predicted_char_index = np.argmax(predictions)

        return predicted_char_index

    def save_cropped_images(self, output_directory, my_array, bounding_boxes):
        for i, (x_orig, y_orig, w, h) in enumerate(bounding_boxes):
            # Choose the first scaled coordinate for each bounding box
            x_scaled, y_scaled = my_array[i]

            # Crop the character from the grayscale image
            char_image = self.image[y_orig:y_orig + h, x_orig:x_orig + w]

            # Use the original and scaled coordinates for saving
            char_output_filename = os.path.join(output_directory,
            #f'char_{i}_x{x_orig}_{x_scaled}_y{y_orig}_{y_scaled}.png')
            f'char_{i}_x{x_orig}_y{y_orig}.png')
            cv2.imwrite(char_output_filename, char_image)

            print(f"Cropped image saved as {char_output_filename}")
    @staticmethod
    def get_image_dimensions(image_path):
        with PILImage.open(image_path) as img:
            width, height = img.size
        return width, height

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

    def preprocess_image_and_save(self, char_image, output_path, x, y):
        # Resize the image to 28x28
        img = PILImage.fromarray(char_image)
        img = img.resize((28, 28))

        # Save the character image to the output directory with modified coordinates
        img.save(output_path)

        # Return the path to the saved character image
        return output_path


    def is_point_inside_rectangle(self, point, rectangle_origin, rectangle_width, rectangle_height):
        x, y = point
        x_rect, y_rect = rectangle_origin
        rect_right = x_rect + rectangle_width
        rect_bottom = y_rect + rectangle_height

        return x_rect <= x <= rect_right and y_rect <= y <= rect_bottom

    def convert_hex_to_rgb(hex_color):
        try:
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4, 6))
        except ValueError as ve:
            print(f"Error converting hex to RGB: {ve}, hex_color: {hex_color}")
            return None
    def determine_font_size(self, h):
        if 80 <= h < 90:
            font_size = 22
        elif 90 <= h < 100:
            font_size = 24
        elif 100 <= h < 110:
            font_size = 26
        elif 110 <= h < 120:
            font_size = 28
        elif 120 <= h < 130:
            font_size = 28
        elif 130 <= h < 140:
            font_size = 30
        elif 140 <= h < 150:
            font_size = 32
        elif 150 <= h < 160:
            font_size = 34
        elif 160 <= h < 170:
            font_size = 36
        else:
            font_size = 26
        return font_size


    @staticmethod
    def group_adjacent_characters(char_images_with_coordinates):
        print("Inside group_adjacent_characters method")
        char_groups = []
        current_group = []

        char_images_with_coordinates.sort(key=lambda char_info: char_info[1])

        some_threshold = 10

        for i in range(len(char_images_with_coordinates) - 1):
            current_char_info = char_images_with_coordinates[i]
            next_char_info = char_images_with_coordinates[i + 1]

            # Compare x-coordinates instead of y-coordinates
            if abs(current_char_info[0][1][0] - next_char_info[0][1][0]) < some_threshold:
                current_group.append(current_char_info)
            else:
                current_group.append(current_char_info)
                char_groups.append(current_group)
                current_group = []

        current_group.append(char_images_with_coordinates[-1])
        char_groups.append(current_group)

        print("Char groups:", char_groups)
        return char_groups

    def organize_characters_in_pdf(self, output_pdf_path, char_groups, input_image_path):
        try:
            print("Inside organize_characters_in_pdf method")
            img_width, img_height = self.get_image_dimensions(input_image_path)
            pdf = canvas.Canvas(output_pdf_path, pagesize=(img_width, img_height))

            for group in char_groups:
                # Sort characters within the group based on their X coordinates
                group.sort(key=lambda char: char[0][1][0])
                print(f"Debug: Group - {group}")
                print(f"Debug: First character coordinates - {group[0][1]}")

                adjacent_chars = []
                for char_info in group:
                    print(f"Debug: char_info - {char_info}")
                    print(f"Debug: Length of char_info - {len(char_info)}")
                    try:
                        # Handle variations in the structure of char_info
                        if len(char_info) == 3:
                            char_output_filename, predicted_char_index, color = char_info
                            x, y, w, h = 0, 0, 0, 0  # You might need to adjust these values based on your requirements
                        elif len(char_info) == 4:
                            char_output_filename, (x, y, w, h), predicted_char_index, color = char_info
                        else:
                            raise ValueError("Unexpected structure for char_info")

                        # Convert variables to integers
                        x, y, w, h = map(int, (x, y, w, h))

                        # Check if the current character is adjacent to the previous one
                        if adjacent_chars and x - adjacent_chars[-1][1] <= adjacent_chars[-1][3]:
                            adjacent_chars.append((char_output_filename, x, y, w, h, predicted_char_index, color))
                        else:
                            # Draw adjacent characters together
                            self.draw_adjacent_characters(pdf, adjacent_chars, img_height)
                            # Reset variables for the new adjacent group
                            adjacent_chars = [(char_output_filename, x, y, w, h, predicted_char_index, color)]

                    except Exception as e:
                        print(f"An error occurred while processing char_info: {char_info}")
                        print(f"Exception details: {e}")

                # Draw the last set of adjacent characters
                self.draw_adjacent_characters(pdf, adjacent_chars, img_height)

            pdf.save()

        except Exception as e:
            print(f"An error occurred while organizing characters in PDF: {e}")

    def draw_adjacent_characters(self, pdf, char_info_list, img_height):
        print("Inside draw_adjacent_characters method")
        print(f"Debug: Length of char_info_list - {len(char_info_list)}")

        # Check if char_info_list is not empty
        if not char_info_list:
            return

        # Determine font size based on character height
        font_size = self.determine_font_size(char_info_list[0][3])  # Use the height of the first character

        # Sort char_info_list based on x-coordinate
        char_info_list.sort(key=lambda char_info: char_info[1])

        # Dictionary to store x-positions for each character group
        x_positions = {}

        for char_info in char_info_list:
            print("Drawing text at coordinates:", char_info[1:])
            char_output_filename, x, y, w, h, predicted_char_index, color = char_info

            # Extract color information based on your implementation
            hex_color = color.replace("ff", "")
            hex_color = hex_color.replace("(", "")
            hex_color = hex_color.replace(")", "")
            color = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            r, g, b = color

            # Set font and color
            pdf.setFillColorRGB(r / 255, g / 255, b / 255)
            pdf.setFont("Helvetica", font_size)

            # Invert y-coordinate to match ReportLab's coordinate system
            inverted_y = img_height - (y + h + 10)
            print(f"Debug: Original y: {y}, Inverted y: {inverted_y}")
            # Check if character is part of an adjacent group
            if predicted_char_index in x_positions:
                # Character is in an adjacent group, use x-coordinate from the first character of the group
                x_position = x_positions[predicted_char_index]
                print(f"Debug: Drawing adjacent character at x-position {x_position}")

                # Draw the character horizontally
                pdf.drawString(x_position, inverted_y, f"{predicted_char_index}")

                # Update x-position for the character group
                x_positions[predicted_char_index] += w  # Use width to create a standard space
            else:
                # Character is not in an adjacent group, use its own x-coordinate

                # Draw the character at its original x, y coordinates
                pdf.drawString(x, inverted_y, f"{predicted_char_index}")

                # Update x-position for the character
                x_positions[predicted_char_index] = x + w  # Use width to create a standard space

    def save_to_pdf(self, output_pdf_path, char_images_with_coordinates, input_image_path, slide_num, slide_count):
        try:
            img_width, img_height = self.get_image_dimensions(input_image_path)
            pdf = canvas.Canvas(output_pdf_path, pagesize=(img_width, img_height))
            print("first")
            print(slide_num)
            print("Length of char_images_with_coordinates:", len(char_images_with_coordinates))
            print("Contents of char_images_with_coordinates:", char_images_with_coordinates)

            if slide_num <= slide_count:
                pdf.showPage()
                print("new page ")
            # Group adjacent characters
            char_groups = self.group_adjacent_characters(char_images_with_coordinates)
            # Organize characters in the PDF
            self.organize_characters_in_pdf(output_pdf_path, char_groups,input_image_path)


        except Exception as e:

            import traceback

            traceback.print_exc()

            print(f"An error occurred while organizing characters in PDF: {e}")


    def change_background_and_text_color(self, input_image_path, output_image_path, output_directory, my_array,slide_number,Slide_count):
        try:
            # scale the point received from API
            scalledarry = []
            for i in my_array:
                x1 = i.x
                y1 = i.y
                original_width, original_height = 342.4285714285, 730.1428571428571
                new_width, new_height = self.get_image_dimensions(input_image_path)

                x2, y2 = scale_point(x1, y1, original_width, original_height, new_width, new_height)
                scalledarry.append(StrokeObject(x2, y2, i.thickness, i.color))

            # Load the image using OpenCV
            self.image = cv2.imread(input_image_path)

            if self.image is None:
                raise Exception("Image not found.")

            # Convert the image to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

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
                roi = binary[y:y + h, x:x + w]
                self.image[y:y + h, x:x + w][roi == 255] = [255, 255, 255]  # White

            # Set the background to black
            self.image[binary == 0] = [0, 0, 0]

            # Save the processed image
            cv2.imwrite(output_image_path, self.image)

            print("Image processed successfully and saved as", output_image_path)

            # Now, process the image to extract characters with bounding boxes
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                 11, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
            dilated_image = cv2.dilate(cleaned_image, None, iterations=2)

            contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes = []

            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Adjust the area threshold as needed
                    (x, y, w, h) = cv2.boundingRect(contour)

                    # Filter based on aspect ratio
                    aspect_ratio = w / float(h)
                    if 0.5 <= aspect_ratio <= 2.0:  # Adjust the aspect ratio range as needed
                        bounding_boxes.append((x, y, w, h))

            bounding_boxes.sort(key=lambda x: (x[1], x[0]), reverse=True)
            # Inside the loop
            for i, (x, y, w, h) in enumerate(bounding_boxes):
                char_image = gray_image[y:y + h, x:x + w]

                # Find the corresponding scaled point for the current bounding box
                for stroke_object in my_array:
                    x_orig, y_orig = stroke_object.x, stroke_object.y
                    x_scaled, y_scaled = scale_point(x_orig, y_orig, original_width, original_height, new_width,new_height)
                    rectangle_origin = (x, y)
                    rectangle_width = w
                    rectangle_height = h
                    point_to_check = (x_scaled, y_scaled)

                    if self.is_point_inside_rectangle(point_to_check, rectangle_origin, rectangle_width,rectangle_height):
                        print("Point is inside the rectangle.")
                        pos = stroke_object
                        # Use your model to predict the character index (modify this part based on your model)
                        predicted_char_index = self.predict_character_index(char_image)

                        # Save the image with both original and scaled coordinates in the filename
                        char_output_filename = os.path.join(output_directory,
                                                            f'char_{i}_orig_x{x_orig}_y{y_orig}_scaled_x{x_scaled}_y{y_scaled}_predicted{predicted_char_index}.png')
                        print(
                            f"x_orig: {x_orig}, y_orig: {y_orig}, x_scaled: {x_scaled}, y_scaled: {y_scaled}, predicted_char_index: {predicted_char_index}")
                        cv2.imwrite(char_output_filename, char_image)

                        char_images_with_coordinates.append(((char_output_filename, (x_orig, y_orig, w, h)),predicted_char_index,pos.color))

                        break

                    # Save the processed image with bounding boxes
                    output_image_with_boxes_path = os.path.join(output_directory,f'output_image_with_boxes_slide_{slide_number}.jpg')

                # Draw bounding boxes on the image
                image_with_boxes = self.image.copy()
                for (x, y, w, h) in bounding_boxes:
                    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Save the image with bounding boxes
                cv2.imwrite(output_image_with_boxes_path, image_with_boxes)


        except Exception as e:
                print("An error occurred:", str(e))

def scale_point(x, y, original_width, original_height, new_width, new_height):
        x_scaled = int((x / original_width) * new_width)
        y_scaled = int((y / original_height) * new_height)
        return x_scaled, y_scaled
# Set the upload folder path
app.config['UPLOAD_FOLDER'] = 'C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/getAllData', methods=['POST'])
def getAllData():
    try:
        data = request.get_json()
        if isinstance(data, str):
            data = json.loads(data)

        # Extract necessary fields
        teacher_id = data.get('TeacherID')
        section_id = data.get('SectionID')
        title = data.get('Title')
        course = data.get('Course')
        lecture = data.get('Lecture')
        images = data.get('Images', [])
        all_segments = data.get('AllSegments', [])
        # print("Received data:", teacher_id, section_id, title, course, lecture)

        # Define output_directory in the outer scope
        output_directory = None

        if not teacher_id or not section_id or not title or not lecture or not course or not images:
            missing_fields = []
            if not teacher_id:
                missing_fields.append("TeacherID")
            if not section_id:
                missing_fields.append("SectionID")
            if not title:
                missing_fields.append("Title")
            if not lecture:
                missing_fields.append("Lecture")
            if not course:
                missing_fields.append("Course")
            if not images:
                missing_fields.append("Images")

            return jsonify({"error": f"Missing required data: {', '.join(missing_fields)}"}), 400

        # Check if TeacherID and SectionID are valid foreign keys
        connection = connect_to_database()
        cursor = connection.cursor()

        try:
            cursor.execute("SELECT TeacherID FROM Teacher WHERE TeacherID = ?", (teacher_id,))
            if cursor.fetchone() is None:
                connection.close()
                return jsonify({"error": "Invalid TeacherID"}), 400

            cursor.execute("SELECT SectionID FROM Section WHERE SectionID = ?", (section_id,))
            if cursor.fetchone() is None:
                connection.close()
                return jsonify({"error": "Invalid SectionID"}), 400

            # Get the current system date
            current_date = datetime.datetime.now().date()

            # Insert into Document table
            cursor.execute(
                "INSERT INTO Document (TeacherID, SectionID, Title, Date, Course, Lecture) VALUES (?, ?, ?, ?, ?, ?)",
                (teacher_id, section_id, title, current_date, course, lecture))
            connection.commit()

            # Get the DocumentID of the inserted row
            cursor.execute("SELECT @@IDENTITY")
            document_id = cursor.fetchone()[0]

            # Adjusted path to have ProjectData inside document_id folder
            output_directory = os.path.join(app.config['UPLOAD_FOLDER'], str(document_id), 'ProjectData')
            os.makedirs(output_directory, exist_ok=True)

            for sequence, base64_image in enumerate(images, start=1):
                image_bytes = base64.b64decode(base64_image)
                image = Image.open(BytesIO(image_bytes))

                # Convert the image to RGB mode
                image = image.convert('RGB')

                # Save the image to the server
                filename = f"image_{sequence}.jpg"
                image_path = os.path.join(output_directory, filename)
                print(f"Saving image to: {image_path}")  # Add this line
                image.save(image_path)

                # Insert image path and sequence into Image table
                cursor.execute("INSERT INTO Image (DocumentID, ImagePath, Sequence) VALUES (?, ?, ?)",
                               (document_id, image_path, sequence))
                connection.commit()

            organized_data_dict = {}
            # Counter variable to store the count of images
            Slide_count = 0
            # Inside the loop where images are saved
            for i, base64_image in enumerate(images):
                slide_number = i
                segments_for_slide = [segment for segment in all_segments if segment['slideNumber'] == slide_number]
                Slide_count += 1

                # Save the image to the server
                filename = f"{title.replace(' ', '_')}_{lecture}_image_{i + 1}.jpg"  # Use 'i' as the sequence
                image_path = os.path.join(output_directory, filename)  # Use output_directory

                try:
                    image = Image.open(BytesIO(base64.b64decode(base64_image)))
                    # Convert the image to RGB mode if needed
                    image = image.convert('RGB')
                    # Save the image in JPG format
                    image.save(image_path, format='JPEG')
                except Exception as e:
                    print(f"Error saving image for slide {slide_number + 1}: {e}")

                if slide_number in organized_data_dict:
                    organized_data_dict[slide_number]['segments'].extend(segments_for_slide)
                else:
                    organized_data_dict[slide_number] = {
                        'slideNumber': slide_number + 1,
                        'image': image_path,
                        'segments': segments_for_slide
                    }

            organized_data = list(organized_data_dict.values())

            char_recognizer = CharacterRecognition(model_path)
            print("Total Slide")
            print(Slide_count)
            pdf_identifier = f'{course}_{lecture}'.replace(" ", "_").lower()
            output_pdf_path = os.path.join(output_directory, f'{pdf_identifier}.pdf')

            for slide_data in organized_data:
                slide_number = slide_data['slideNumber']
                image_path = slide_data['image']
                char_segments = slide_data.get('segments', [])
                my_array = []

                for segment in char_segments:
                    # Check the content of the segment dictionary
                    x = segment['startPoint']['dx']
                    y = segment['startPoint']['dy']
                    thickness = segment['thickness']
                    color = segment['color']

                    duplicate_segment = next(
                        (item for item in my_array if
                        item.x == x and item.y == y and item.thickness == thickness and item.color == color),
                        None)

                    if not duplicate_segment:
                        my_array.append(StrokeObject(x, y, thickness, color))

                print(
                    f"Processing Slide {slide_number} - Image Path: {image_path}, Output Directory: {output_directory}, Segments: {my_array}")
                char_recognizer.change_background_and_text_color(image_path, output_image_path, output_directory, my_array, slide_number, Slide_count)
                char_recognizer.save_to_pdf(output_pdf_path, char_images_with_coordinates, image_path, slide_number, Slide_count)
            # Update the DocumentPath column in the Document table
            cursor.execute("UPDATE Document SET DocumentPath = ? WHERE DocumentID = ?", (output_pdf_path, document_id))
            connection.commit()
            print(f"Processed characters saved in '{output_directory}' directory.")
            print(f"Predicted characters and their positions saved in '{output_pdf_path}'.")
            # Update the DocumentPath column in the Document table
        finally:
            connection.close()

        response_message = {'status': 'success', 'message': 'Data received and processed successfully'}
        return jsonify(response_message), 200

    except Exception as e:
        error_message = {'status': 'error', 'message': f'Error processing data: {str(e)}'}
        return jsonify(error_message), 500

if __name__ == '__main__':
    model_path = 'C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Work\\mnist_model.h5'
    output_image_path = 'C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\output_image.jpg'

    app.run(host='0.0.0.0', port=5000, debug=False)