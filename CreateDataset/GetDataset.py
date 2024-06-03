import cv2
import numpy as np
import glob
import csv
np.set_printoptions(threshold=9999)

def average_points(points):
    # Step 1: Create a dictionary where the keys are the x-coordinates and the values are lists of y-coordinates
    x_dict = {}
    for x, y in points:
        if x not in x_dict:
            x_dict[x] = []
        x_dict[x].append(y)

    # Step 2: For each key in the dictionary, if there is more than one y-coordinate, calculate the average y-coordinate
    for x in x_dict:
        if len(x_dict[x]) > 1:
            x_dict[x] = sum(x_dict[x]) / len(x_dict[x])

    # Step 3: Convert the dictionary back into a list of tuples
    return [(x, y) for x, y in x_dict.items()]
y_value_list = []
x_vals = []

image_files = glob.glob("GetDataFromVids/BottomRightPS/FrameCrops/*.png")
for file in image_files:
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  # Reads in image file
    # Apply thresholding to obtain a binary image
    _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    # Invert the binary image
    binary_image = cv2.bitwise_not(binary_image)
    # Apply the binary mask to extract the graph and set the background to white
    contrasted_image = np.where(binary_image == 255, image, 255)
    contrasted_image = cv2.flip(contrasted_image, 0)
    cv2.imwrite("new.jpg",contrasted_image)
    indices = np.where(contrasted_image < 235)  # Returns 2 arrays of y and x coordinates, respectively, of each pixel that is NOT white (255)
    points = list((zip(indices[1], indices[0])))  # Zips up x array and y array to make coordinate list
    # Sort the list of tuples in place by the first element of each tuple
    points.sort(key=lambda x: x[0])

    # Initialize an empty list to hold the updated points
    updated_points = []

    # Iterate through the sorted points
    for i in range(len(points) - 1):
        # Add the current point to the updated list
        updated_points.append(points[i])

        # Check if there is a gap between the current and next x values
        if points[i + 1][0] - points[i][0] > 1:
            # Calculate the y value of the new point as the y value of the previous point
            y_value = points[i][1]

            # Create new points with the x value of the gap and the y value of the previous point
            for x in range(points[i][0] + 1, points[i + 1][0]):
                updated_points.append((x, y_value))

    # Add the last point to the updated list
    updated_points.append(points[-1])

    avgpoints = average_points(updated_points)

    # Convert list of ints in tuples to int
    i = 0
    while (i < len(avgpoints)):
        if isinstance(avgpoints[i][1], list):
            x = list(avgpoints[i])
            x[1] = x[1][0]
            avgpoints[i] = tuple(x)
        i += 1

    # AVGPOINTS IS THE LIST OF POINTS ON THE GRAPH
    # Following section will normalize the points on the y-axis and convert the x-values to wavenumber values
    x_step = (4000-725)/2100 # step value between each x val = (total change in wavevalue over the entire x-axis)/(# of x-axis pixels)
    # Reverse the order of x values
    x_flipped_points = [(len(avgpoints) + 2 - x, y) for x, y in avgpoints]

    # Normalize y and x values
    updated_points = [(round((x*x_step)+725, 5), round(y/284,10)) for x, y in x_flipped_points]

    # Separate x and y values into two lists
    x_values = [x for x, y in updated_points]
    y_values = [y for x, y in updated_points]
    y_value_list.append(y_values)

# Open the CSV file in write mode
with open('Data_PS_New.csv', 'a', newline='') as file:
    writer = csv.writer(file)

    # Write x values to the first line
    # writer.writerow(x_vals)

    # Write y values to the other lines
    for item in y_value_list:
        writer.writerow(item)
