from flask import Flask, render_template, request
import sqlite3
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

model = tf.keras.models.load_model('fine_tuned_CNN.h5',compile=False)


def analyse(path):
    img_height = 180
    img_width = 180
   
    img = tf.keras.utils.load_img(
        path, target_size=(img_height, img_width)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return score

def calculate_shapes(image_path):
    image = Image.open(image_path)

    image_array = np.array(image)

    if len(image_array.shape) == 3:
        image_array = np.mean(image_array, axis=-1)

    threshold = 128
    binary_image = (image_array > threshold).astype(np.uint8)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_shapes = len(contours)

    return num_shapes    

def identify_Affected_Area(image_path, img):
    image = cv2.imread(image_path)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([140, 50, 180])
    upper_pink = np.array([170, 255, 255])

    pink_mask = cv2.inRange(hsv_image, lower_pink, upper_pink)

    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    affected_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > affected_area:
            affected_area = area
            largest_contour = contour

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi = image[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
        image[y:y+h, x:x+w] = blurred_roi

        cv2.imwrite('static/blur/'+img, image)

    return affected_area

def count_affected_lymph(image_path):
    image = cv2.imread(image_path)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_pink = np.array([140, 50, 180])
    upper_pink = np.array([170, 255, 255])

    pink_mask = cv2.inRange(hsv_image, lower_pink, upper_pink)

    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    num_abnormalities = 0
    if largest_contour is not None:
        for contour in contours:
            if contour is not largest_contour:
                intersection_area = cv2.contourArea(cv2.convexHull(contour))
                if intersection_area > 0:
                    num_abnormalities += 1

    return num_abnormalities

def get_true_label(image_name):
    true_label = image_name[0].lower()
    return true_label
    
app = Flask(__name__)

@app.route('/evaluate_model', methods=['GET', 'POST'])
def evaluate_model():
    if request.method == 'POST':
        test_data_path = "static\\test"

        true_labels = []
        predicted_labels = []
        class_names=['b', 'g']
        accuracy_values = []
        Run=1
        for _ in range(Run):
            for image_name in os.listdir(test_data_path):
                image_path = os.path.join(test_data_path, image_name)
                true_label = get_true_label(image_name)
                true_labels.append(true_label)

                predicted_score = analyse(image_path)
                predicted_label = class_names[np.argmax(predicted_score)]
                predicted_labels.append(predicted_label)

            accuracy = accuracy_score(true_labels, predicted_labels)
            accuracy_values.append(accuracy)
            print(f"Accuracy: {accuracy * 100:.2f}%")

        confusion_mat = confusion_matrix(true_labels, predicted_labels)

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

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1_score:.2f}")

        history_df = pd.read_csv('training_history.csv')
        error_rate = 1 - history_df['accuracy']


        #Plot training & validation loss values
        plt.plot(history_df['loss'])
        plt.plot(history_df['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        #plt.show()
        plt.savefig('static/model_loss.png')
        plt.close()

        # Plot training & validation accuracy values
        plt.plot(history_df['accuracy'])
        plt.plot(history_df['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        #plt.show()
        plt.savefig('static/model_accuracy.png')
        plt.close()

        # Plot error rate graph
        plt.plot(error_rate)
        plt.title('Error Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')
        plt.savefig('static/error_rate.png')
        #plt.show()

        confusion_matrix_dimensions = confusion_mat.shape
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
            model_accuracy='model_accuracy.png',
            model_loss="model_loss.png",
            error_rate="error_rate.png",
            error_rate_values = error_rate.tolist(),
            bg="bgImage.png"
        )
    return render_template('evaluation_result.html', bg="bgImage.png")  # Or redirect to the appropriate page

@app.route('/')
def index():
    return render_template('index.html', bg="bgImage.png")

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
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again',bg="bgImage.png")
        else:
            return render_template('userlog.html', bg="bgImage.png")

    return render_template('index.html', bg="bgImage.png")


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

        return render_template('index.html', msg='Successfully Registered', bg="bgImage.png")
    
    return render_template('index.html', bg="bgImage.png")

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
            identify_Affected_Area(path, image)
            images.append("http://127.0.0.1:5000/static/blur/"+image)

            num_abnormalities = count_affected_lymph(path)
            print(f"Number of lymph nodes in the metastasis area: {num_abnormalities}")
            results.append(f"Number of affected lymph nodes: {num_abnormalities}")
            
        if class_names[np.argmax(score)]=='good':
            identify_Affected_Area(path, image)
            images.append("http://127.0.0.1:5000/static/blur/"+image)


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

        #Counting number of cells
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_cells = len(contours)
        if total_cells < 100:
            total_cells*=90
        elif total_cells >=100:
            total_cells*=50
        print(f"Total number of cells: {total_cells}")
        results.append(f"Total number of cells: {total_cells}")

        # Overlap detection
        overlap_cells = 0
        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                # Calculate centroids
                M_i = cv2.moments(contours[i])
                M_j = cv2.moments(contours[j])

                # Ensure non-zero area before calculating centroids
                if M_i['m00'] != 0 and M_j['m00'] != 0:
                    cx_i = int(M_i['m10'] / M_i['m00'])
                    cy_i = int(M_i['m01'] / M_i['m00'])

                    cx_j = int(M_j['m10'] / M_j['m00'])
                    cy_j = int(M_j['m01'] / M_j['m00'])

                    # Calculate distance between centroids
                    distance = np.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)

                    if distance < 20:
                        overlap_cells += 1
        print(f"Total number of overlapped cells: {overlap_cells}")
        results.append(f"Total number of overlapped cells: {overlap_cells}")

        if class_names[np.argmax(score)]!='good':
            # Damaged cell detection
            damaged_cells = 0
            min_area_threshold = 100
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area_threshold:
                    damaged_cells += 1
            print(f"Total number of damaged cells: {damaged_cells}")
            results.append(f"Total number of damaged cells: {damaged_cells}")

        return render_template('userlog.html', images=images, results=results, bg="bgImage.png")

    return render_template('userlog.html')

@app.route('/logout')
def logout():
    return render_template('index.html', bg="bgImage.png")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
