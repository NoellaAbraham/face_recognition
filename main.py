import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import datetime

window_name = 'Attendance Management'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)


#button positions for registration
button_x1, button_y1, button_width1, button_height1 = 350, 450, 200, 50

#button positions for detecting emotions
button_x2, button_y2, button_width2, button_height2 = 600, 450, 100, 50


# Registration button click event handler
def register_button_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Capture image
        image_path = 'employee_images'  # Specify the path to save registration images
        os.makedirs(image_path, exist_ok=True)

        image_name = input("Enter image name: ")  # Allow the user to specify the image name
        image_name = f"{image_path}/{image_name}.jpg"

        cv2.imwrite(image_name, frame)  # Save the captured image

        print(f"Image saved as {image_name}")


def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

path = 'employee_images'  # Path that contains all the employee pictures
studentImg = []
studentName = []
myList = os.listdir(path)
for cl in myList:
    curimg = cv2.imread(f'{path}/{cl}')
    studentImg.append(curimg)
    studentName.append(os.path.splitext(cl)[0])  # Getting the name of the employee from the name the image is saved as

def findEncoding(images):
    imgEncodings = []
    for img in images:
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings

EncodeList = findEncoding(studentImg)

vid = cv2.VideoCapture(0)  # Use 0 for the default webcam

cv2.setMouseCallback(window_name, register_button_click)

background_path = 'bg.png'
background = cv2.imread(background_path)

while True:
    success, frame = vid.read()

    # Resize the frame to a smaller size
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Adjust the scaling factor

     # Create a copy of the background image
    result = background.copy()

    facesInFrame = face_rec.face_locations(small_frame)
    encodeFacesInFrame = face_rec.face_encodings(small_frame, facesInFrame)


    # Calculate the position to place the small frame within the result image
    top = 190  # top position
    left = 5  # left position

    # Overlay the small frame onto the result image at the specified position
    result[top:top + small_frame.shape[0], left:left + small_frame.shape[1]] = small_frame

    font_scale = 0.7 #font size of button

    # Draw the registration button rectangle
    
    cv2.rectangle(result, (button_x1, button_y1), (button_x1 + button_width1, button_y1 + button_height1), (253, 156, 80) , -1)

    # Draw the detect button rectangle
    cv2.rectangle(result, (button_x2, button_y2), (button_x2 + button_width2, button_y2 + button_height2), (253, 156, 80), -1)

    # Calculate the size of the text for the registration button
    text_size, _ = cv2.getTextSize("Registration", cv2.FONT_HERSHEY_COMPLEX, font_scale, 1)

    #Calculate the size of the text for the detect button

    text_size2, _ = cv2.getTextSize("Detect", cv2.FONT_HERSHEY_COMPLEX, font_scale, 1)


    # Calculate the position to center the text on the registration button
    text_x1 = button_x1 + (button_width1 - text_size[0]) // 2
    text_y1 = button_y1 + (button_height1 + text_size[1]) // 2
    
    # Calculate the position to center the text on the detect button
    text_x2 = button_x2 + (button_width2 - text_size2[0]) // 2
    text_y2 = button_y2 + (button_height2 + text_size2[1]) // 2

    

    # Draw the text on the button
    cv2.putText(result, "Registration", (text_x1, text_y1), cv2.FONT_HERSHEY_COMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)

    cv2.putText(result, "Detect", (text_x2, text_y2), cv2.FONT_HERSHEY_COMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)

    # Comparing the face from live webcam to the one in the picture
    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame):
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        print(facedis)
        matchIndex = np.argmin(facedis)

        # If face in live webcam matches the one in the employee picture
        if matches[matchIndex]:
            name = studentName[matchIndex].upper()

            y1, x2, y2, x1 = faceloc
            font_size = 0.5


            y1, x2, y2, x1 = y1 * 1, x2 * 1, y2 * 1, x1 * 1  # Adjust the multiplier as needed
            cv2.rectangle(result, (left + x1, top + y1), (left + x2, top + y2+20), (0, 255, 0), 2)  # Draw the bounding box on the result image
            cv2.rectangle(result, (left + x1, top+20 + y2 - 20), (left + x2, top+20 + y2), (0, 255, 0), cv2.FILLED)  # Draw the filled rectangle for the text background
            cv2.putText(result, name, (left + x1 + 3, top+20 + y2 - 3), cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255), 2)  # Draw the text on the result image


            cv2.putText(result, "Verified!", (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(result, "Hello, "+name, (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y %H:%M:%S")


            

            cv2.putText(result, date_time, (350, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        else:
            y1, x2, y2, x1 = faceloc  # Define variables for bounding box coordinations
            y1, x2, y2, x1 = y1 * 1, x2 * 1, y2 * 1, x1 * 1
            font_size = 0.5
            cv2.rectangle(result, (left + x1, top + y1), (left + x2, top + y2+20), (0, 0, 255), 2)  # Draw the bounding box on the result image
            cv2.rectangle(result, (left + x1, top+20 + y2 - 20), (left + x2, top+20 + y2), (0, 0, 255), cv2.FILLED)  # Draw the filled rectangle for the text background
            cv2.putText(result, "Unknown", (left + x1 + 3, top+20 + y2 - 3), cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 255, 255), 2)  # Draw the text on the result image


            cv2.putText(result, "Unknown Person!", (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            now = datetime.now()
            date_time = now.strftime("%d/%m/%Y %H:%M:%S")


            cv2.putText(result, date_time, (350, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            

    # Display the result in the window
    cv2.imshow(window_name, result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
vid.release()
cv2.destroyAllWindows()
