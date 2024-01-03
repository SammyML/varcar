import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import json
import textwrap

# Load color data from JSON file
with open('Colors.json') as f:
    color_data = json.load(f)


# Callback function for the accuracy scale trackbar
def on_accuracy_change(value):
    global accuracy_scale
    accuracy_scale = value


def get_dominant_color(image):
    image = cv2.resize(image, (50, 50))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    threshold_value = 100
    image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)[1]

    pixels = image.reshape((-1, 3))

    n_colors = 1
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)

    largest_cluster_index = np.argmax(np.bincount(kmeans.labels_))
    dominant_color = list(map(int, kmeans.cluster_centers_[largest_cluster_index]))

    return dominant_color


def find_closest_color(rgb_values, color_data):
    min_distance = float('inf')
    closest_color = None

    for color in color_data["data"]:
        color_code = color["color_code"]
        color_code_rgb = np.array([int(color_code[i:i + 2], 16) for i in (0, 2, 4)])
        current_distance = distance.euclidean(rgb_values, color_code_rgb)

        if current_distance < min_distance:
            min_distance = current_distance
            closest_color = color

    return closest_color


def print_boxed_text(text):
    box_horizontal = "\u2500"  # Unicode character for a horizontal line
    box_vertical = "\u2502"  # Unicode character for a vertical line
    box_top_left = "\u250C"  # Unicode character for the top left corner of a box
    box_top_right = "\u2510"  # Unicode character for the top right corner of a box
    box_bottom_left = "\u2514"  # Unicode character for the bottom left corner of a box
    box_bottom_right = "\u2518"  # Unicode character for the bottom right corner of a box

    lines = textwrap.wrap(text, width=40)

    max_line_length = max(len(line) for line in lines)
    box_width = max_line_length + 2  # Width of the box
    box_height = len(lines) + 2  # Height of the box

    print(f"{box_top_left}{box_horizontal * box_width}{box_top_right}")

    for line in lines:
        print(f"{box_vertical} {line.ljust(max_line_length)} {box_vertical}")

    print(f"{box_bottom_left}{box_horizontal * box_width}{box_bottom_right}")



def main():
    global accuracy_scale
    accuracy_scale = 50  # Initial accuracy scale value

    # Create a window
    cv2.namedWindow("Color Detection App", cv2.WINDOW_NORMAL)

    # Create a trackbar for the accuracy scale
    cv2.createTrackbar("Accuracy Scale", "Color Detection App", accuracy_scale, 100, on_accuracy_change)

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Get dominant color
        dominant_color = get_dominant_color(frame)

        # Find the closest color in the dataset based on RGB values
        closest_color = find_closest_color(dominant_color, color_data)

        # Display the webcam feed
        cv2.imshow("Color Detection App", frame)

        # Display predicted color information
        print("Predicted Color:")
        print(f"RGB Values: {dominant_color}")

        # Display accuracy
        print(f"Prediction Accuracy: {accuracy_scale}%")

        # Display color information in the console within a box
        if closest_color is not None:
            info_text = f"Color Information:\n" \
                        f"Color ID: {closest_color['color_id']}\n" \
                        f"Color Code: {closest_color['color_code']}\n" \
                        f"Color Name: {closest_color['color_name']}\n" \
                        f"Color Type: {closest_color['color_type']}"
            print_boxed_text(info_text)
        else:
            print_boxed_text("Color information not found for the detected color.")

        # Break the loop when 'esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
