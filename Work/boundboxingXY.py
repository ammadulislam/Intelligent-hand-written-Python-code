import cv2
import os

# Input image file path
input_image_path = r"C:\Users\HP\PycharmProjects\Fyp_Final\Data\1337\ProjectData\fgg_Lecture_1_2_image_1.jpg"  # Replace with your image file path
output_directory = 'output'  # Output directory where processed images will be saved

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load the image using OpenCV
image = cv2.imread(input_image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to preprocess the image
thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Use morphology operations to clean up the image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
cleaned_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

# Find contours on the cleaned image
contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store bounding boxes along with their starting points
bounding_boxes = []

# Loop through the contours and filter out small ones
for contour in contours:
    if cv2.contourArea(contour) > 50:  # Adjust the area threshold as needed
        (x, y, w, h) = cv2.boundingRect(contour)
        bounding_boxes.append(((x, y), (x + w, y + h)))

# Sort the bounding boxes by their y-coordinate and then by x-coordinate
bounding_boxes.sort(key=lambda box: (box[0][1], box[0][0]))

# Initialize a variable to store the previous end_x
prev_end_x = None

# Loop through the sorted bounding boxes and draw them on the original image
for i, ((start_x, start_y), (end_x, end_y)) in enumerate(bounding_boxes):
    # Draw a bounding box around the character on the original image
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Add text annotation with the starting point (x, y) and sequence number on the image
    sequence_number = i + 1
    annotation_text = f"{sequence_number} ({start_x}, {start_y})"
    cv2.putText(image, annotation_text, (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Crop the character from the grayscale image
    char_image = gray_image[start_y:end_y, start_x:end_x]

    # Save the character image to the output directory
    output_filename = os.path.join(output_directory, f'char_{sequence_number}.png')
    cv2.imwrite(output_filename, char_image)

    # Calculate and print the width difference
    if prev_end_x is not None:
        width_difference = start_x - prev_end_x
        print(f"Width difference between bounding boxes {i} and {i - 1}: {width_difference}")

    # Update prev_end_x
    prev_end_x = end_x

# Save the original image with bounding boxes
output_image_path = os.path.join(output_directory, 'output_image_with_boxes.jpg')
cv2.imwrite(output_image_path, image)

# Display the image
cv2.imshow("Image with Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Processed characters saved in '{output_directory}' directory.")
print(f"Original image with bounding boxes saved as '{output_image_path}'.")












# Inside the loop for drawing words on the PDF
            # for i, word in enumerate(words):
            #     print(f"Wordsssssssss {i + 1}:")
            #     print(i)
            #     print(word)
            #
            #     # Get the starting position of the first character in the word
            #     first_char_x, first_char_y, _, _ = word[0][1]
            #
            #     # Calculate the horizontal position based on the character index
            #     char_width = 8  # Adjust as needed
            #     char_spacing = 3
            #
            #     # Determine font size for the word based on the first character's height
            #     first_char_height = word[0][1][3]
            #     if 80 <= first_char_height < 90:
            #         font_size = 22
            #     elif 90 <= first_char_height < 100:
            #         font_size = 24
            #     elif 100 <= first_char_height < 110:
            #         font_size = 26
            #     elif 110 <= first_char_height < 120:
            #         font_size = 28
            #     elif 120 <= first_char_height < 130:
            #         font_size = 28
            #     else:
            #         font_size = 26
            #
            #     # Invert y-coordinate to match ReportLab's coordinate system
            #     inverted_y = img_height - (first_char_y + first_char_height + 10)
            #
            #     for j, (predicted_char_index, (x, y, w, h)) in enumerate(word):
            #         if isinstance(color, tuple):  # Check if color is already a tuple
            #             r, g, b = color
            #         else:
            #             hex_color = color.replace("ff", "")
            #             hex_color = hex_color.replace("(", "")
            #             hex_color = hex_color.replace(")", "")
            #             r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            #
            #         # Draw the word on the PDF at the specified coordinates with the determined font size and color
            #         pdf.setFillColorRGB(r / 255, g / 255, b / 255)
            #         pdf.setFont("Helvetica", font_size)
            #
            #         # Calculate the horizontal position based on the character index
            #         char_x_position = first_char_x + (char_width + char_spacing) * j
            #
            #         # Display only the predicted_char_index
            #         pdf.drawString(char_x_position, inverted_y, str(predicted_char_index))
            #
            #     inverted_y -= 10  # Adjust as needed
            #
            # pdf.save()





























            #correct adjacent finder

#
# "C:\Program Files\Python311\python.exe" C:\Users\HP\PycharmProjects\Fyp_Final\Newone.py
# Connection to SQL Server successful
#  * Serving Flask app 'Newone'
#  * Debug mode: off
# WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
#  * Running on all addresses (0.0.0.0)
#  * Running on http://127.0.0.1:5000
#  * Running on http://192.168.1.9:5000
# Press CTRL+C to quit
# [{'id': 1, 'startPoint': {'dx': 80.71428571428572, 'dy': 192.42857142857144}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 2, 'startPoint': {'dx': 121.85714285714286, 'dy': 187.28571428571428}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 3, 'startPoint': {'dx': 157.85714285714286, 'dy': 187.8571428571429}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 4, 'startPoint': {'dx': 89.28571428571428, 'dy': 340.42857142857144}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 5, 'startPoint': {'dx': 144.71428571428572, 'dy': 355.8571428571429}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 6, 'startPoint': {'dx': 155.57142857142858, 'dy': 353.0}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 7, 'startPoint': {'dx': 96.14285714285714, 'dy': 527.2857142857142}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 8, 'startPoint': {'dx': 132.71428571428572, 'dy': 530.7142857142858}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 9, 'startPoint': {'dx': 161.28571428571428, 'dy': 535.2857142857142}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 10, 'startPoint': {'dx': 186.42857142857142, 'dy': 538.1428571428571}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 11, 'startPoint': {'dx': 41.85714285714286, 'dy': 649.5714285714287}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 12, 'startPoint': {'dx': 67.57142857142858, 'dy': 649.5714285714287}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 13, 'startPoint': {'dx': 98.42857142857142, 'dy': 649.5714285714287}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 14, 'startPoint': {'dx': 257.85714285714283, 'dy': 663.8571428571429}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 15, 'startPoint': {'dx': 293.2857142857143, 'dy': 665.5714285714287}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}, {'id': 16, 'startPoint': {'dx': 320.7142857142857, 'dy': 668.4285714285713}, 'color': 'ff000000', 'thickness': 4, 'slideNumber': 0}]
# Saving image to: C:\Users\HP\PycharmProjects\Fyp_Final\Data\1434\ProjectData\image_1.jpg
# Total Slide
# 1
# Processing Slide 1 - Image Path: C:\Users\HP\PycharmProjects\Fyp_Final\Data\1434\ProjectData\bh_Lecture_1_2_image_1.jpg, Output Directory: C:\Users\HP\PycharmProjects\Fyp_Final\Data\1434\ProjectData, Segments: [<__main__.StrokeObject object at 0x000002E5F0D41510>, <__main__.StrokeObject object at 0x000002E5F0D78290>, <__main__.StrokeObject object at 0x000002E5F0D7AB10>, <__main__.StrokeObject object at 0x000002E5F118F910>, <__main__.StrokeObject object at 0x000002E5F118FB10>, <__main__.StrokeObject object at 0x000002E5F266E890>, <__main__.StrokeObject object at 0x000002E5F266F690>, <__main__.StrokeObject object at 0x000002E5F266DE10>, <__main__.StrokeObject object at 0x000002E5F266CE10>, <__main__.StrokeObject object at 0x000002E5F266CB50>, <__main__.StrokeObject object at 0x000002E5F117D590>, <__main__.StrokeObject object at 0x000002E5F117C8D0>, <__main__.StrokeObject object at 0x000002E5F10E4150>, <__main__.StrokeObject object at 0x000002E5F10E5E10>, <__main__.StrokeObject object at 0x000002E5F10E5FD0>, <__main__.StrokeObject object at 0x000002E5F10E4D10>]
# Image processed successfully and saved as C:\Users\HP\PycharmProjects\Fyp_Final\output_image.jpg
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 248ms/step
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 37ms/step
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 56ms/step
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 50ms/step
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 44ms/step
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 41ms/step
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 30ms/step
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 34ms/step
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 48ms/step
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 42ms/step
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 54ms/step
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 30ms/step
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 30ms/step
# Point is inside the rectangle.
# 1/1 [==============================] - 0s 33ms/step
# char images with coordinates [(('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_0_orig_x320_y668_scaled_x561_y1169_predicted7.png', (552, 1154), (320, 668, 42, 77)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_1_orig_x293_y665_scaled_x513_y1164_predicted7.png', (504, 1152), (293, 665, 48, 58)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_2_orig_x257_y663_scaled_x451_y1161_predicted7.png', (442, 1151), (257, 663, 53, 61)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_3_orig_x98_y649_scaled_x172_y1136_predicted7.png', (163, 1127), (98, 649, 61, 88)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_4_orig_x41_y649_scaled_x73_y1136_predicted8.png', (64, 1127), (41, 649, 96, 99)), 8, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_5_orig_x186_y538_scaled_x326_y941_predicted7.png', (317, 928), (186, 538, 46, 63)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_6_orig_x161_y535_scaled_x282_y936_predicted7.png', (273, 925), (161, 535, 42, 72)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_7_orig_x132_y530_scaled_x232_y928_predicted5.png', (217, 919), (132, 530, 55, 91)), 5, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_8_orig_x96_y527_scaled_x168_y922_predicted3.png', (159, 912), (96, 527, 59, 102)), 3, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_9_orig_x144_y355_scaled_x253_y622_predicted7.png', (199, 609), (144, 355, 111, 88)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_10_orig_x89_y340_scaled_x156_y595_predicted7.png', (138, 586), (89, 340, 59, 116)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_11_orig_x157_y187_scaled_x276_y328_predicted3.png', (262, 316), (157, 187, 59, 91)), 3, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_12_orig_x121_y187_scaled_x213_y327_predicted3.png', (202, 313), (121, 187, 56, 91)), 3, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_13_orig_x80_y192_scaled_x141_y336_predicted2.png', (117, 308), (80, 192, 69, 101)), 2, 'ff000000')]
# after Sort [(('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_13_orig_x80_y192_scaled_x141_y336_predicted2.png', (117, 308), (80, 192, 69, 101)), 2, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_12_orig_x121_y187_scaled_x213_y327_predicted3.png', (202, 313), (121, 187, 56, 91)), 3, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_11_orig_x157_y187_scaled_x276_y328_predicted3.png', (262, 316), (157, 187, 59, 91)), 3, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_10_orig_x89_y340_scaled_x156_y595_predicted7.png', (138, 586), (89, 340, 59, 116)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_9_orig_x144_y355_scaled_x253_y622_predicted7.png', (199, 609), (144, 355, 111, 88)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_8_orig_x96_y527_scaled_x168_y922_predicted3.png', (159, 912), (96, 527, 59, 102)), 3, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_7_orig_x132_y530_scaled_x232_y928_predicted5.png', (217, 919), (132, 530, 55, 91)), 5, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_6_orig_x161_y535_scaled_x282_y936_predicted7.png', (273, 925), (161, 535, 42, 72)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_5_orig_x186_y538_scaled_x326_y941_predicted7.png', (317, 928), (186, 538, 46, 63)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_4_orig_x41_y649_scaled_x73_y1136_predicted8.png', (64, 1127), (41, 649, 96, 99)), 8, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_3_orig_x98_y649_scaled_x172_y1136_predicted7.png', (163, 1127), (98, 649, 61, 88)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_2_orig_x257_y663_scaled_x451_y1161_predicted7.png', (442, 1151), (257, 663, 53, 61)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_1_orig_x293_y665_scaled_x513_y1164_predicted7.png', (504, 1152), (293, 665, 48, 58)), 7, 'ff000000'), (('C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Data\\1434\\ProjectData\\char_0_orig_x320_y668_scaled_x561_y1169_predicted7.png', (552, 1154), (320, 668, 42, 77)), 7, 'ff000000')]
# Line 1:
#   Words in Line 1:
#     Word 1:
# [(2, (80, 192, 69, 101), 'ff000000'), (3, (121, 187, 56, 91), 'ff000000'), (3, (157, 187, 59, 91), 'ff000000')]
# Line 2:
#   Words in Line 2:
#     Word 1:
# [(7, (89, 340, 59, 116), 'ff000000'), (7, (144, 355, 111, 88), 'ff000000')]
# Line 3:
#   Words in Line 3:
#     Word 1:
# [(3, (96, 527, 59, 102), 'ff000000'), (5, (132, 530, 55, 91), 'ff000000'), (7, (161, 535, 42, 72), 'ff000000'), (7, (186, 538, 46, 63), 'ff000000')]
# Line 4:
#   Words in Line 4:
#     Word 1:
# [(8, (41, 649, 96, 99), 'ff000000'), (7, (98, 649, 61, 88), 'ff000000')]
#     Word 2:
# [(7, (257, 663, 53, 61), 'ff000000'), (7, (293, 665, 48, 58), 'ff000000'), (7, (320, 668, 42, 77), 'ff000000')]
# Total number of lines: 4
# Processed characters saved in 'C:\Users\HP\PycharmProjects\Fyp_Final\Data\1434\ProjectData' directory.
# Predicted characters and their positions saved in 'C:\Users\HP\PycharmProjects\Fyp_Final\Data\1434\ProjectData\pf_lecture_1_2.pdf'.
# 192.168.1.7 - - [20/Dec/2023 22:10:44] "POST /getAllData HTTP/1.1" 200 -

