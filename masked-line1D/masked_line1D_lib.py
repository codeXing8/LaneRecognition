# Dependencies
import cv2
import numpy as np
import time
import warnings

# Parameters
LINE_COLOR = (255, 0, 0)
BLUR_KERNEL = (5, 5)
CANNY_MIN = 50
CANNY_MAX = 150

# ROI mask
TRIANGLE_MASK_BOTTOM_END_FACTOR = 1
TRIANGLE_MASK_LEFT_END_FACTOR = 0.19
TRIANGLE_MASK_RIGHT_END_FACTOR = 0.82
TRIANGLE_MASK_TOP_WIDTH_FACTOR = 0.48
TRIANGLE_MASK_TOP_HEIGHT_FACTOR = 0.47

# Line filtering
UPPER_COORDINATE_LIMIT = 900
LOWER_COORDINATE_LIMIT = -900
IMG_DIM_SUM_FACTOR_STRAIGHT_LINE_LEFT_UPPER = 1.0
IMG_DIM_SUM_FACTOR_STRAIGHT_LINE_LEFT_LOWER = 0.88
IMG_DIM_SUM_FACTOR_STRAIGHT_LINE_RIGHT_UPPER = 1.34
IMG_DIM_SUM_FACTOR_STRAIGHT_LINE_RIGHT_LOWER = 1.22

# Averaged line coordinates gradient filter settings
AVERAGED_LINE_X_LOWER_GRADIENT_LIMIT = -30
AVERAGED_LINE_X_UPPER_GRADIENT_LIMIT = 30

# Compensate lane loss and store last lines
VALID_LAST_LINES_ARRAY_LENGTH = 3

# Canny filter pre-processing
def pre_process_canny(img):
    # Apply filters
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, BLUR_KERNEL, 0)
    img_canny = cv2.Canny(img_blur, CANNY_MIN, CANNY_MAX)
    return img_canny


# Extract rectangular ROI
def mask_rectangular_roi(img, color):
    height = img.shape[0]
    width = img.shape[1]
    # tri = np.array([[(200, height), (1100, height), (550, 250)]])
    tri = np.array([[(int(TRIANGLE_MASK_LEFT_END_FACTOR * width), int(TRIANGLE_MASK_BOTTOM_END_FACTOR * height)),
                     (int(TRIANGLE_MASK_RIGHT_END_FACTOR * width), int(TRIANGLE_MASK_BOTTOM_END_FACTOR * height)),
                     (int(TRIANGLE_MASK_TOP_WIDTH_FACTOR * width), int(TRIANGLE_MASK_TOP_HEIGHT_FACTOR * height))]])
    # print(tri)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, tri, color)
    img_mask = cv2.bitwise_and(img, mask)
    return img_mask


# Filter and decide on lane characteristics
def filter_lines(lines, img_dim_sum):
    valid_lines = []
    line_id = []
    if lines is not None:
        for line in lines:
            # print(line)
            coordinates = line.reshape(4)
            valid_line = True
            coordinates_sum = 0
            for i in range(len(coordinates)):
                coordinates_sum += coordinates[i]
                if coordinates[i] > UPPER_COORDINATE_LIMIT or coordinates[i] < LOWER_COORDINATE_LIMIT:
                    valid_line = False
            if valid_line:
                valid_lines.append(line)

            # Calc line identifying values
            id_straight_left_up = img_dim_sum * IMG_DIM_SUM_FACTOR_STRAIGHT_LINE_LEFT_UPPER
            id_straight_left_low = img_dim_sum * IMG_DIM_SUM_FACTOR_STRAIGHT_LINE_LEFT_LOWER
            id_straight_right_up = img_dim_sum * IMG_DIM_SUM_FACTOR_STRAIGHT_LINE_RIGHT_UPPER
            id_straight_right_low = img_dim_sum * IMG_DIM_SUM_FACTOR_STRAIGHT_LINE_RIGHT_LOWER

            # Identify lines:
            # 0 - unknown
            # 1 - straight line, left side
            # 2 - straight line, right side
            print(line)
            if valid_line and id_straight_left_low < coordinates_sum < id_straight_left_up:
                line_id.append(1)
            elif valid_line and id_straight_right_low < coordinates_sum < id_straight_right_up:
                line_id.append(2)
            elif valid_line:
                line_id.append(0)
    return [np.array(line_id), np.array(valid_lines)]


# Based on line ID, change lane markings color
def line_id2color(line_id):
    if line_id == 0:
        return 0, 0, 255
    elif line_id == 1:
        return 100, 255, 0
    elif line_id == 2:
        return 255, 100, 0
    else:
        return 0, 0, 0


# Plot lines on lanes
def display_lines(img, lines, line_id):
    img_line = np.zeros(np.append(np.array(img.shape), 3), dtype=int).astype(np.uint8)
    line_cnt = 0
    color_array = []
    for line in lines:
        x1, y1, x2, y2 = np.around(line.reshape(4))
        # cv2.line(img_line, (x1, y1), (x2, y2), LINE_COLOR, 10)
        color_array.append(line_id2color(line_id[line_cnt]))
        cv2.line(img_line, (x1, y1), (x2, y2), color_array[line_cnt], 10)
        line_cnt += 1
    return img_line


# Sort edge recognition results, get average of results as single final result
def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    detected_line_type = 0
    if lines is None:
        # detected_line_type: 0 none, 1 pair, 2 left only, 3 right only
        # left_line = []
        # right_line = []
        return [detected_line_type, np.array([])]
    else:
        # Log
        line_cnt = 0
        for line in lines:
            # Log
            line_cnt += 1
            # print("Line No.", line_cnt, "of", len(lines))

            # Turn line coordinates into line parameters
            x1, y1, x2, y2 = line.reshape(4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            # Log
            # print("Slope:", slope, "y-intercept:", intercept)
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        if len(left_fit) == 0 and len(right_fit) == 0:
            # left_line = []
            # right_line = []
            return [detected_line_type, np.array([])]
        elif len(left_fit) == 0:
            detected_line_type = 3
            right_fit_avg = np.average(right_fit, axis=0)
            right_line = make_coordinates(img, right_fit_avg)
            left_line = []
            return [detected_line_type, np.array([right_line])]
        elif len(right_fit) == 0:
            detected_line_type = 2
            left_fit_avg = np.average(left_fit, axis=0)
            left_line = make_coordinates(img, left_fit_avg)
            right_line = []
            return [detected_line_type, np.array([left_line])]
        else:
            detected_line_type = 1
            left_fit_avg = np.average(left_fit, axis=0)
            right_fit_avg = np.average(right_fit, axis=0)
            left_line = make_coordinates(img, left_fit_avg)
            right_line = make_coordinates(img, right_fit_avg)
            return [detected_line_type, np.array([left_line, right_line])]


def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1 * 3/5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


# limit_gradient(averaged_lines[0][0], averaged_lines_last[0][0], averaged_lines_init[0][0])
def limit_gradient(current_value, last_value):
    lower_limit, upper_limit = AVERAGED_LINE_X_LOWER_GRADIENT_LIMIT, AVERAGED_LINE_X_UPPER_GRADIENT_LIMIT
    if current_value - last_value < lower_limit:
        new_value = last_value + lower_limit
    elif current_value - last_value > upper_limit:
        new_value = last_value + upper_limit
    else:
        new_value = current_value
    return new_value


# Append valid line to storage function
def append_valid_line_to_storage(averaged_lines, valid_last_lines):
    if len(valid_last_lines) < VALID_LAST_LINES_ARRAY_LENGTH:
        valid_last_lines.append(averaged_lines)
    else:
        # Only store last lines up to a number of VALID_LAST_LINES_ARRAY_LENGTH
        # If longer, shift stored lanes and pop highest entry - then add newest
        valid_last_lines_tmp = []
        for i in range(len(valid_last_lines) - 1):
            valid_last_lines_tmp.append(valid_last_lines[i + 1])
        valid_last_lines_tmp.append(averaged_lines)
        valid_last_lines = valid_last_lines_tmp
    # Return valid lines
    return valid_last_lines


# Calc average coordinates along stored lines
def calc_average_coordinates_along_stored_lines(valid_last_lines):
    # Init coordinate sum arrays for average calculation
    sum_left, sum_right = np.asarray([0, 0, 0, 0]), np.asarray([0, 0, 0, 0])
    # Iterate over stored lines
    for valid_last_line in valid_last_lines:
        # Separate left and right lines
        valid_last_left_line, valid_last_right_line = valid_last_line[0], valid_last_line[1]
        sum_left, sum_right = sum_left + valid_last_left_line, sum_right + valid_last_right_line
    # Calc average coordinates of last stored lines for left and right separately
    avg_left, avg_right = np.array(sum_left / len(valid_last_lines)).astype(int), np.array(sum_right / len(valid_last_lines)).astype(int)
    averaged_lines = [avg_left, avg_right]
    line_id = np.array([1, 2])
    return line_id, averaged_lines