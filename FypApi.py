
import traceback
import cv2
import os
import numpy as np

import tensorflow as tf
from flask import  send_file
from PIL import Image as PILImage
from reportlab.pdfgen import canvas
from flask import request
import datetime
from reportlab.platypus import  Image
from flask import Flask, jsonify
from flask_cors import CORS
from Test import StrokeObject
app = Flask(__name__)
CORS(app)
import base64
from PIL import Image
from io import BytesIO
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
def get_current_time():
    return datetime.datetime.now()
import pyodbc

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








#                                                                                                                        login

@app.route('/login', methods=['POST'])
def login():
    data = request.json

    if not data or 'Email' not in data or 'Password' not in data:
        return jsonify({"error": "Invalid login data"}), 400

    email = data['Email']
    password = data['Password']

    # Retrieve user data from the database based on the provided username
    connection = connect_to_database()
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT UserID, Password, Role FROM Users WHERE Email = ?", (email,))
        user_data = cursor.fetchone()

        if user_data is None:
            return jsonify({"message": "User not found"}), 404

        stored_password = user_data[1]  # Index 1 corresponds to the 'Password' column
        user_role = user_data[2]  # Index 2 corresponds to the 'Role' column
        user_id = user_data[0]  # Index 0 corresponds to the 'UserID' column

        if password == stored_password:
            return jsonify({"message": "Login successful", "Role": user_role, "UserID": user_id}), 200
        else:
            return jsonify({"message": "Invalid password"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()










#                                                                                                     Signup


# A list to store user data as dictionaries
users = []
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json

    if not data or 'Email' not in data or 'Password' not in data or 'Role' not in data:
        return jsonify({"message": "Invalid request data"}), 400

    email = data['Email']
    password = data['Password']
    role = data['Role']



    print("Received data:", username, password, role)  # Add this line for debugging



    # Check if the user already exists
    for user in users:
        if user['Username'] == username:
          return jsonify({"message": "User already exists"}), 400

    # Create a connection
    connection = connect_to_database()
    cursor = connection.cursor()

    # Execute SQL to insert user data
    cursor.execute("INSERT INTO Users (Email, Password, Role) VALUES (?, ?, ?)",
                   (email, password, role))

    # Commit changes and close the connection
    connection.commit()
    connection.close()

    # Retrieve the UserID of the newly registered user
    connection = connect_to_database()
    cursor = connection.cursor()
    cursor.execute("SELECT UserID FROM Users WHERE Email = ?", (email,))
    user_id = cursor.fetchone()[0]  # Get the UserID

    return jsonify({"message": "User registered successfully", "UserID": user_id}), 201

#------------------------------------------------------------------------------------------------------------------------StudentSignUp

@app.route('/Signupstudent', methods=['POST'])
def Signupstudent():
    data = request.json

    if not data or 'UserID' not in data or 'RegistrationNo' not in data or 'StudentName' not in data or 'Discipline' not in data or 'SectionID' not in data:
        error_response = {
            "message": "Invalid request data",
            "details": "Required fields are missing"
        }
        return jsonify({"message": "Invalid request data"}), 400

    user_id = data['UserID']
    registration_no = data['RegistrationNo']
    student_name = data['StudentName']
    discipline = data['Discipline']
    section_id = data['SectionID']

    print("Received data:", user_id, registration_no, student_name,discipline,section_id)

    connection = connect_to_database()
    cursor = connection.cursor()

    # Check if the user and section exist before adding the student
    cursor.execute("SELECT UserID FROM Users WHERE UserID = ?", (user_id,))
    user_result = cursor.fetchone()
    if user_result is None:
        return jsonify({"message": "User with specified UserID does not exist"}), 400

    cursor.execute("SELECT SectionID FROM Section WHERE SectionID = ?", (section_id,))
    section_result = cursor.fetchone()
    if section_result is None:
        return jsonify({"message": "Section with specified SectionID does not exist"}), 400

    cursor.execute("INSERT INTO Student (UserID, RegistrationNo, StudentName, Discipline, SectionID) VALUES (?, ?, ?, ?, ?)",
                   (user_id, registration_no, student_name, discipline, section_id))

    connection.commit()
    connection.close()

    return jsonify({"message": "Student added successfully"}), 201


#-------------------------------------------------------------------------------------------------------------TeacherSignUp

@app.route('/SignUpteacher', methods=['POST'])
def SignUpteacher():
    data = request.json

    if not data or 'UserID' not in data or 'TeacherName' not in data or 'Designation' not in data or 'PhoneNo' not in data or 'Experience' not in data:
        error_response = {
            "message": "Invalid request data",
            "details": "Required fields are missing"
        }
        return jsonify(error_response), 400

    user_id = data['UserID']
    teacher_name = data['TeacherName']
    designation = data['Designation']
    phone_no = data['PhoneNo']
    experience = data['Experience']

    # Create a connection
    connection = connect_to_database()
    cursor = connection.cursor()

    # Check if the provided UserID exists in the User table
    cursor.execute("SELECT COUNT(*) FROM Users WHERE UserID = ?", (user_id,))
    user_count = cursor.fetchone()[0]

    if user_count == 0:
        error_response = {
            "message": "User with provided UserID does not exist",
            "details": f"UserID {user_id} does not exist"
        }
        return jsonify(error_response), 400

    # Insert the teacher data into the Teacher table
    cursor.execute("INSERT INTO Teacher (UserID, TeacherName, Designation, PhoneNo, Experience) VALUES (?, ?, ?, ?, ?)",
                   (user_id, teacher_name, designation, phone_no, experience))

    # Commit changes and close the connection
    connection.commit()
    connection.close()

    return jsonify({"message": "Teacher added successfully"}), 201





@app.route("/getpdfss")
def getpdfss():
    pdf_name = request.args.get('path')
    print("***********************", pdf_name)

    if pdf_name is None:
        return "Please provide a valid path parameter in the URL", 400

    try:
        return send_file(pdf_name, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404





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
        pdf_list = [{"document_id": record[0], "pdf_path": record[1], "Title": record[2]} for record in pdf_records]

        return jsonify({"pdfs": pdf_list})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An error occurred while processing your request"}), 500


########################################################################################
########################################################################################
#########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################











def get_current_time():
    return datetime.datetime.now()

char_images_with_coordinates = []
class StrokeObject:
    def __init__(self, x, y, thickness, color):
        self.x = x
        self.y = y
        self.thickness= thickness
        self.color = color

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
            # print("Image width",width)
            # print("image height",height)
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

    def determine_font_size(self, first_char_height):
        # Implement this method to determine font size based on the character's height
        if 80 <= first_char_height < 90:
            return 12
        elif 90 <= first_char_height < 100:
            return 16
        elif 100 <= first_char_height < 110:
            return 20
        elif 110 <= first_char_height < 120:
            return 24
        elif 120 <= first_char_height < 130:
            return 28
        elif 130 <= first_char_height < 140:
            return 32
        elif 140 <= first_char_height < 150:
            return 36
        elif 150 <= first_char_height < 160:
            return 40
        elif 160 <= first_char_height < 170:
            return 44
        elif 170 <= first_char_height < 180:
            return 48
        elif 180 <= first_char_height < 190:
            return 52
        elif 190 <= first_char_height < 200:
            return 56
        else:
            return 24

    def extract_rgb_from_hex(self, hex_color):
        hex_color = hex_color.replace("ff", "")
        hex_color = hex_color.replace("(", "")
        hex_color = hex_color.replace(")", "")
        color_tuple = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return color_tuple

    def save_to_pdf(self, output_pdf_path, char_images_with_coordinates, input_image_path, slide_num, slide_count):
        try:
            print("char images with coordinates", char_images_with_coordinates)
            img_width, img_height = self.get_image_dimensions(input_image_path)

            pdf = canvas.Canvas(output_pdf_path, pagesize=(img_width, img_height))
            threshold = 60
            words = []
            current_line = []  # Keep track of characters within a line
            lines = []  # Keep track of lines containing words

            # Sort characters line by line
            char_images_with_coordinates.sort(
                key=lambda char: (char[0][1][1], char[0][1][0], char[0][2][0]))  # Sorting based on (k, j, x)
            print("after Sort", char_images_with_coordinates)
            for i, ((filename, (j, k), (x, y, w, h)), predicted_char_index, color) in enumerate(
                    char_images_with_coordinates):
                if current_line:
                    last_x, last_y, _, _ = current_line[-1][1]
                    if y > last_y + threshold:  # Check vertical adjacency
                        # If the current character is not vertically adjacent to the last one,
                        # start a new line
                        lines.append(current_line)
                        current_line = []

                current_line.append((predicted_char_index, (x, y, w, h), color))

            # Add the last line
            current_line = sorted(current_line, key=lambda x: x[1][0])
            if current_line:
                lines.append(current_line)
            words = []  # Keep track of words within a line
            for line_index, line in enumerate(lines):
                print(f"Line {line_index + 1}:")


                current_word = []  # Keep track of characters within a word

                for character in line:
                    if current_word:
                        last_x, _, _, _ = current_word[-1][1]
                        if character[1][0] > last_x + threshold:  # Check horizontal adjacency
                            # If the current character is not horizontally adjacent to the last one,
                            # start a new word
                            words.append(current_word)
                            current_word = []

                    current_word.append(character)
                current_word = sorted(current_word, key=lambda x: x[1][0])
                # Add the last word
                if current_word:
                    words.append(current_word)

                print(f"  Words in Line {line_index + 1}:")
                for word_index, word in enumerate(words):
                    print(f"    Word {word_index + 1}:")
                    print(word)

            for i, word in enumerate(words):
                first_char_x, first_char_y, _, _ = word[0][1]
                char_width = 8
                char_spacing = 6

                first_char_height = word[0][1][3]
                font_size = self.determine_font_size(first_char_height)

                inverted_y = img_height - (first_char_y + first_char_height + 10)

                for j, (predicted_char_index, (x, y, w, h), color) in enumerate(word):
                    if isinstance(color, tuple):
                        r, g, b = color
                    else:
                        r, g, b = self.extract_rgb_from_hex(color)

                    pdf.setFillColorRGB(r / 255, g / 255, b / 255)
                    pdf.setFont("Helvetica", font_size)

                    char_x_position = first_char_x + (char_width + char_spacing) * j
                    pdf.drawString(char_x_position, inverted_y, str(predicted_char_index))

                inverted_y -= 10

            pdf.save()


        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()

    def change_background_and_text_color(self, input_image_path, output_image_path, output_directory, my_array,slide_number,Slide_count):
        try:
            # scale the point received from API
            scalledarry = []
            for i in my_array:
                x1 = i.x
                y1 = i.y
                #samsung Mobile
                # original_width, original_height = 342.4285714285, 730.1428571428571
                #note Mobile
                original_width, original_height =342.42857142857144,675.8571428571429

                new_width, new_height = self.get_image_dimensions(input_image_path)
                # print("Image width/////////////", new_width)
                # print("image height////////////", new_height)

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
                j, k, w, h = cv2.boundingRect(contour)
                roi = binary[k:k + h, j:j + w]
                self.image[k:k + h, j:j + w][roi == 255] = [255, 255, 255]  # White

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
            predicted_characters = []
            # Inside the loop where you process contours

            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Adjust the area threshold as needed
                    (j, k, w, h) = cv2.boundingRect(contour)
                    bounding_boxes.append((j, k, w, h))

            bounding_boxes.sort(key=lambda x: (x[1], x[0]), reverse=True)

            # Inside the loop
            for i, (j, k, w, h) in enumerate(bounding_boxes):
                # Draw a bounding box around the character on the original image
                cv2.rectangle(self.image, (j, k), (j + w, k + h), (0, 255, 0), 2)
                # Crop the character from the grayscale image
                char_image = gray_image[k:k + h, j:j + w]

                #char_image_resized = cv2.resize(char_image, (28, 28))

                # Save the character image to the output directory with bounding box coordinates
                # char_output_filename = os.path.join(output_directory, f'char_{i}_x{j}_y{k}.png')
                # cv2.imwrite(char_output_filename, char_image)
                # Initialize image_with_boxes before the loop

                image_with_boxes = self.image.copy()

                # Find the corresponding scaled point for the current bounding box
                for stroke_object in my_array:
                    x_orig, y_orig = stroke_object.x, stroke_object.y
                    x_scaled, y_scaled = scale_point(x_orig, y_orig, original_width, original_height, new_width,
                                                     new_height)
                    rectangle_origin = (j, k)
                    rectangle_width = w
                    rectangle_height = h
                    point_to_check = (x_scaled, y_scaled)

                    if self.is_point_inside_rectangle(point_to_check, rectangle_origin, rectangle_width,
                                                      rectangle_height):
                        print("Point is inside the rectangle.")
                        pos = stroke_object

                        # Use your model to predict the character index (modify this part based on your model)
                        predicted_char_index = self.predict_character_index(char_image)

                        # Convert x_orig and y_orig to integers
                        x_orig_int, y_orig_int = int(x_orig), int(y_orig)

                        # Save the image with both original and scaled coordinates in the filename
                        char_output_filename = os.path.join(output_directory,
                                                            f'char_{i}_orig_x{x_orig_int}_y{y_orig_int}_scaled_x{x_scaled}_y{y_scaled}_predicted{predicted_char_index}.png')

                        # Annotate the bounding box with starting coordinates (j, k)
                        cv2.putText(image_with_boxes, f'({x_orig_int}, {y_orig_int})', (j, k - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        cv2.imwrite(char_output_filename, char_image)

                        char_images_with_coordinates.append(
                            ((char_output_filename,(j,k),(x_orig_int, y_orig_int, w, h)), predicted_char_index,
                             pos.color))

                        break
            # Save the processed image with bounding boxes
            output_image_with_boxes_path = os.path.join(output_directory,
                                                        f'output_image_with_boxes_slide_{slide_number}.jpg')

            # Draw bounding boxes and starting coordinates on the image
            for (j, k, w, h) in bounding_boxes:
                cv2.rectangle(image_with_boxes, (j, k), (j + w, k + h), (0, 255, 0), 2)

            # Save the image with bounding boxes and starting coordinates
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
        UserID = data.get('UserID')
        section_id = data.get('SectionID')
        title = data.get('Title')
        course = data.get('Course')
        lecture = data.get('Lecture')
        images = data.get('Images', [])
        all_segments = data.get('AllSegments', [])
        print(all_segments)
        print("Received data:", UserID, section_id, title, course, lecture)

        # Define output_directory in the outer scope
        output_directory = None

        if not UserID or not section_id or not title or not lecture or not course or not images:
            missing_fields = []
            if not UserID:
                missing_fields.append("UserID")
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
            cursor.execute("SELECT TeacherID FROM Teacher WHERE UserID = ?", (UserID,))
            result=cursor.fetchone()
            if result is None:
                connection.close()
                return jsonify({"error": "Invalid TeacherID"}), 400
            teacher_id=result[0]
            print("Teacher id  ",teacher_id)


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
            output_pdf_path = os.path.join(output_directory, f'{title}.pdf')
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
    model_path = r'C:\Users\HP\PycharmProjects\Fyp_Final\Work\mnist_model.h5'
    output_image_path = 'C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\output_image.jpg'
    app.run(host='0.0.0.0', port=5000, debug=False)




#############################################################################################################################
#############################################################################################################################
#############################################################################################################################


#
#
# Press CTRL+C to quit
# *********************** C:\Users\HP\PycharmProjects\Fyp_Final\Data\2530\ProjectData\Ecom.pdf
# 192.168.1.8 - - [11/Jan/2024 15:13:05] "GET /getpdfss?path=C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\2530\\ProjectData\\Ecom.pdf HTTP/1.1" 200 -
# *********************** C:\Users\HP\PycharmProjects\Fyp_Final\Data\2530\ProjectData\Ecom.pdf
# 127.0.0.1 - - [11/Jan/2024 15:13:48] "GET /getpdfss?path=C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\2530\\ProjectData\\Ecom.pdf HTTP/1.1" 304 -
# *********************** C:\Users\HP\PycharmProjects\Fyp_Final\Data\2530\ProjectData\Opp.pdf
# 127.0.0.1 - - [11/Jan/2024 15:14:39] "GET /getpdfss?path=C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\2530\\ProjectData\\Opp.pdf HTTP/1.1" 404 -
# *********************** C:\Users\HP\PycharmProjects\Fyp_Final\Data\2529\ProjectData\Opp.pdf
# 127.0.0.1 - - [11/Jan/2024 15:15:06] "GET /getpdfss?path=C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\2529\\ProjectData\\Opp.pdf HTTP/1.1" 200


# http://127.0.0.1:5000/getpdfss?path=C:\Users\HP\PycharmProjects\Fyp_Final\Data\2530\ProjectData\Ecom.pdf
