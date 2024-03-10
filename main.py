from flask import Flask, render_template, url_for, request
import sqlite3
import numpy as np
from PIL import Image
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

model = tf.keras.models.load_model('fine_tuned_CNN.h5',compile=False)


def analyse(path):
    batch_size = 32
    img_height = 180
    img_width = 180
   
    img = tf.keras.utils.load_img(
        path, target_size=(img_height, img_width)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return score

def calculate_shapes(image_path):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Convert the image to grayscale if it's a color image
    if len(image_array.shape) == 3:
        image_array = np.mean(image_array, axis=-1)

    # Threshold the image to get a binary image
    threshold = 128  # Adjust the threshold as needed
    binary_image = (image_array > threshold).astype(np.uint8)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the number of shapes (contours)
    num_shapes = len(contours)

    return num_shapes    

def calculate_largest_blur_area(image_path, img):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for light pink color
    lower_pink = np.array([140, 50, 180])  # Adjust these values based on your specific shade of pink
    upper_pink = np.array([170, 255, 255])

    # Create a mask for the pink color
    pink_mask = cv2.inRange(hsv_image, lower_pink, upper_pink)

    # Find contours in the mask
    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the largest contour (area)
    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    # Blur the largest area in the original image
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi = image[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
        image[y:y+h, x:x+w] = blurred_roi

        # Save the result
        cv2.imwrite('static/blur/'+img, image)

    return largest_area

def calculate_largest_area_in_mm(image_path, pixel_to_mm_conversion):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for light pink color
    lower_pink = np.array([140, 50, 180])  # Adjust these values based on your specific shade of pink
    upper_pink = np.array([170, 255, 255])

    # Create a mask for the pink color
    pink_mask = cv2.inRange(hsv_image, lower_pink, upper_pink)

    # Find contours in the mask
    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the largest contour (area)
    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    # Calculate the largest area in square millimeters
    largest_area_mm = largest_area * pixel_to_mm_conversion**2

    return largest_area_mm

def count_abnormalities_in_largest_area(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for light pink color
    lower_pink = np.array([140, 50, 180])  # Adjust these values based on your specific shade of pink
    upper_pink = np.array([170, 255, 255])

    # Create a mask for the pink color
    pink_mask = cv2.inRange(hsv_image, lower_pink, upper_pink)

    # Find contours in the mask
    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the largest contour (area)
    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    # Count the number of abnormalities in the largest area
    num_abnormalities = 0
    if largest_contour is not None:
        for contour in contours:
            # Check if the contour is not the largest contour and intersects with it
            if contour is not largest_contour:
                # Calculate the intersection area
                intersection_area = cv2.contourArea(cv2.convexHull(contour))
                if intersection_area > 0:
                    num_abnormalities += 1

    return num_abnormalities

def get_true_label(image_name):
    # Assuming the label is the first character of the image name
    true_label = image_name[0].lower()

    # For example, if the label is present after a space, you can use:
    # true_label = image_name.split()[1].lower()
    return true_label
    
app = Flask(__name__)

@app.route('/evaluate_model', methods=['GET', 'POST'])
def evaluate_model():
    if request.method == 'POST':
        test_data_path = "static\\test"  # Replace with your actual test dataset path

        true_labels = []
        predicted_labels = []
        class_names=['b', 'g']
        accuracy_values = []
        Run=5
        for _ in range(Run):
            for image_name in os.listdir(test_data_path):
                image_path = os.path.join(test_data_path, image_name)
                true_label = get_true_label(image_name)
                true_labels.append(true_label)

                # Use your 'analyse' function to get model predictions
                predicted_score = analyse(image_path)
                predicted_label = class_names[np.argmax(predicted_score)]
                predicted_labels.append(predicted_label)

            # Calculate accuracy
            accuracy = accuracy_score(true_labels, predicted_labels)
            accuracy_values.append(accuracy)
            print(f"Accuracy: {accuracy * 100:.2f}%")

        # Create confusion matrix
        confusion_mat = confusion_matrix(true_labels, predicted_labels)

        # Extract TP, TN, FP, FN
        TP = confusion_mat[1, 1]
        TN = confusion_mat[0, 0]
        FP = confusion_mat[0, 1]
        FN = confusion_mat[1, 0]

        print("Confusion Matrix:")
        print(confusion_mat)
        print(f"True Positives (TP): {TP}")
        print(f"True Negatives (TN): {TN}")
        print(f"False Positives (FP): {FP}")
        print(f"False Negatives (FN): {FN}")

        # Calculate additional metrics
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1_score:.2f}")

        plt.plot(range(1, len(accuracy_values) + 1), accuracy_values)
        plt.xlabel('Run')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Runs')
        plt.grid(True)
        plt.savefig('static/accuracy_graph.png')  # Save the graph as an image
        plt.close()

        confusion_matrix_dimensions = confusion_mat.shape
        # Render a template or return JSON response with the metrics
        return render_template(
            'evaluation_result.html',
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confusion_matrix=confusion_mat.tolist(),  # Convert NumPy array to list for JSON serialization
            confusion_matrix_dimensions=confusion_matrix_dimensions,
            TP=TP,
            TN=TN,
            FP=FP,
            FN=FN,
            accuracy_graph='accuracy_graph.png'
        )
    return render_template('evaluation_result.html')  # Or redirect to the appropriate page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        image = request.form['img']
        
        path = "static/test/"+image

        images = []
        results = []
        class_names=['bad', 'good']
        score = analyse(path)
        print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
        images.append("http://127.0.0.1:5000/static/test/"+image)
        results.append("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))

        if class_names[np.argmax(score)]!='good':
            #num_shapes = calculate_shapes(path)
            #print(f"Number of shapes: {num_shapes}")
            #results.append(f"Number of shapes: {num_shapes}")

            largest_blur_area = calculate_largest_blur_area(path, image)
            print(f"Area: {largest_blur_area}")
            images.append("http://127.0.0.1:5000/static/blur/"+image)
            results.append(f"Area: {largest_blur_area} mm")

            #pixel_to_mm_conversion = 0.1  # Replace with your pixel-to-mm conversion factor
            #largest_area_mm = calculate_largest_area_in_mm(path, pixel_to_mm_conversion)
            #print(f"Largest area in square millimeters: {largest_area_mm} mm")
            #results.append(f"Largest area in square millimeters: {largest_area_mm} mm")

            num_abnormalities = count_abnormalities_in_largest_area(path)
            print(f"Number of lymph nodes in the metastasis area: {num_abnormalities}")
            results.append(f"Number of lymph nodes in the metastasis area: {num_abnormalities}")

        image1 = cv2.imread(path)
        gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray/'+image, gray_image)
        images.append("http://127.0.0.1:5000/static/gray/"+image)

        #apply the Canny edge detection
        edges = cv2.Canny(image1, 100, 200)
        cv2.imwrite('static/edges/'+image, edges)
        images.append("http://127.0.0.1:5000/static/edges/"+image)

        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold/'+image, threshold2)
        images.append("http://127.0.0.1:5000/static/threshold/"+image)

        return render_template('userlog.html', images=images, results=results)

    return render_template('userlog.html')

@app.route('/logout')
def logout():
    return render_template('index.html')


    

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
