# Import dependencies
import numpy as np
import cv2
import pickle

# Constants
# Yellow and white color filter
DARK_YELLOW = np.array([18, 94, 140])
BRIGHT_YELLOW = np.array([48, 255, 255])
DARK_WHITE = np.array([0, 0, 200])
BRIGHT_WHITE = np.array([255, 255, 255])

# Dilation and erosion, Gaussian blur kernel
DILATE_ERODE_KERNEL = np.ones((5, 5))
GAUSSIAN_BLUR_KERNEL = (5, 5)

# Sliding window filter
MIN_NUMBER_OF_PIXELS_PER_WINDOW = 1
LEFT_LANE_COLOR = [255, 0, 100]
RIGHT_LANE_COLOR = [0, 100, 255]

# Image warp parameters
WARP_POINTS = np.float32([(42 / 100, 63 / 100), (1 - (42 / 100), 63 / 100), (14 / 100, 87 / 100), (1 - (14 / 100), 87 / 100)])
ORIG_POINTS = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])


# Undistort with calibration pickle file function
def undistort_with_cal_pickle_file(img, cal_dir):
    # Open pickle file
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    # Load distortion parameters
    mtx = file['mtx']
    dist = file['dist']
    # Perform un-distortion
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Return un-distorted image
    return dst


# Color filter function
def apply_yellow_white_filter(img):
    # Convert image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Keep white and yellow image parts
    img_masked_white, img_masked_yellow = cv2.inRange(img_hsv, DARK_WHITE, BRIGHT_WHITE), cv2.inRange(img_hsv, DARK_YELLOW, BRIGHT_YELLOW)
    # Unite images
    img_masked = cv2.bitwise_or(img_masked_white, img_masked_yellow)
    # Return image after union
    return img_masked


# Run Canny edge detector in a color masked image function
def run_masked_canny_detector(img):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur filter and Canny edge detection
    img_blurred = cv2.GaussianBlur(img_gray, GAUSSIAN_BLUR_KERNEL, 0)
    img_canny = cv2.Canny(img_blurred, 50, 100)
    # Perform dialation and erosion
    img_dila = cv2.dilate(img_canny, DILATE_ERODE_KERNEL, iterations=1)
    img_eros = cv2.erode(img_dila, DILATE_ERODE_KERNEL, iterations=1)
    # Apply yellow and white color filter
    img_masked = apply_yellow_white_filter(img)
    # Unite edge detection and color filter results
    img_masked_canny = cv2.bitwise_or(img_masked, img_eros)
    # Return united image, Canny detection results and color masked image
    return img_masked_canny, img_canny, img_masked


# Perform image warp function 
def perform_image_warp(img, dst_size, src, dst):
    # Get image size as float
    img_size = np.float32([(img.shape[1],img.shape[0])])
    # Multiply define source and destination points during image warp
    src = src * img_size
    dst = dst * np.float32(dst_size)
    # Determine perspective transformation matrix
    M_trans = cv2.getPerspectiveTransform(src, dst)
    # Perform perspective transformation
    img_warped = cv2.warpPerspective(img, M_trans, dst_size)
    # Return warped image
    return img_warped


# Get image histogram function
def get_image_histogram(img):
    hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
    return hist


# Apply sliding window filter function
def apply_sliding_window_filter(img, n_windows=15, margin=50, draw_windows=True):
    # Prepare output image
    img_out = np.dstack((img, img, img)) * 255

    # Get image histogram - 0 if black and positive if bright color, e. g. yellow or white
    histogram = get_image_histogram(img)
    
    # find peaks of left and right halves
    histogram_midpoint = int(histogram.shape[0] / 2)
    # Find brightness maximum in left half of the histogram
    window_x_left = np.argmax(histogram[:histogram_midpoint])
    # Find brightness maximum in right half of the histogram
    window_x_right = np.argmax(histogram[histogram_midpoint:]) + histogram_midpoint
    # Current positions to be updated for each window
    window_x_left_current, window_x_right_current = window_x_left, window_x_right

    # Set height of windows depending on number of windows along image height
    window_height = np.int(img.shape[0] / n_windows)
    # Identify the x and y positions of all non-zero pixels in the image
    inds_nonzero = img.nonzero()
    inds_y_nonzero, inds_x_nonzero = np.array(inds_nonzero[0]), np.array(inds_nonzero[1])
    
    # Create empty lists to receive left and right lane pixel indices
    inds_left_lane, inds_right_lane = [], []
    # Step through the windows one by one
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left) - according to window height and width (margin)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = window_x_left_current - margin
        win_xleft_high = window_x_left_current + margin
        win_xright_low = window_x_right_current - margin
        win_xright_high = window_x_right_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(img_out, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (100, 255, 255), 1)
            cv2.rectangle(img_out, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (100, 255, 255), 1)
        # Identify the nonzero pixels in x and y within the window
        inds_left_nonzero = ((inds_y_nonzero >= win_y_low) & (inds_y_nonzero < win_y_high) &
                          (inds_x_nonzero >= win_xleft_low) & (inds_x_nonzero < win_xleft_high)).nonzero()[0]
        inds_right_nonzero = ((inds_y_nonzero >= win_y_low) & (inds_y_nonzero < win_y_high) &
                           (inds_x_nonzero >= win_xright_low) & (inds_x_nonzero < win_xright_high)).nonzero()[0]
        # Re-center next window along mean of pixel indizes within current window
        if len(inds_left_nonzero) > MIN_NUMBER_OF_PIXELS_PER_WINDOW:
            window_x_left_current = np.int(np.mean(inds_x_nonzero[inds_left_nonzero]))
        if len(inds_right_nonzero) > MIN_NUMBER_OF_PIXELS_PER_WINDOW:
            window_x_right_current = np.int(np.mean(inds_x_nonzero[inds_right_nonzero]))
        # Append left and right lane points to the list
        inds_left_lane.append(inds_left_nonzero)
        inds_right_lane.append(inds_right_nonzero)
        
    # Concatenate the arrays of indices
    inds_left_lane, inds_right_lane = np.concatenate(inds_left_lane), np.concatenate(inds_right_lane)

    # Get left and right line pixel positions as arrays of points
    points_x_left, points_y_left = inds_x_nonzero[inds_left_lane], inds_y_nonzero[inds_left_lane]
    points_x_right, points_y_right = inds_x_nonzero[inds_right_lane], inds_y_nonzero[inds_right_lane]

    # If both left and right x coordinate point arrays hold points, perform polynomial fit
    if points_x_left.size and points_x_right.size:
        # Fit a second order polynomial to left and right lane point sets
        left_fit = np.polyfit(points_y_left, points_x_left, 2)
        right_fit = np.polyfit(points_y_right, points_x_right, 2)

        # Get polynomial parameters for left and right lane
        left_a, left_b, left_c, right_a, right_b, right_c = [], [], [], [], [], []
        left_a.append(left_fit[0])
        left_b.append(left_fit[1])
        left_c.append(left_fit[2])
        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])
        
        # Use average parameters for actual left and right lane fit
        left_fit, right_fit = np.empty(3), np.empty(3)
        left_fit[0], left_fit[1], left_fit[2] = np.mean(left_a[-10:]), np.mean(left_b[-10:]), np.mean(left_c[-10:])
        right_fit[0], right_fit[1], right_fit[2] = np.mean(right_a[-10:]), np.mean(right_b[-10:]), np.mean(right_c[-10:])

        # Generate x and y values from image shape to plot points along polynomial fit
        point_range = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fit_x = left_fit[0] * point_range ** 2 + left_fit[1] * point_range + left_fit[2]
        right_fit_x = right_fit[0] * point_range ** 2 + right_fit[1] * point_range + right_fit[2]

        # Colorize polynomial lane fit results in output image
        img_out[inds_y_nonzero[inds_left_lane], inds_x_nonzero[inds_left_lane]] = LEFT_LANE_COLOR
        img_out[inds_y_nonzero[inds_right_lane], inds_x_nonzero[inds_right_lane]] = RIGHT_LANE_COLOR

        # Return output image and both lane fit points in x and y as well as range of points along fit
        return img_out, (left_fit_x, right_fit_x), point_range
    # If no lanes where found, return original image
    else:
        # Return original image
        return img, (0,0), (0,0), 0


# Draw lanes function
def draw_lanes(img, left_fit, right_fit, frame_width, frame_height, src):
    # Define point range along image shape
    point_range = np.linspace(0, img.shape[0] - 1, img.shape[0])
    # Prepare output image
    img_color = np.zeros_like(img)
    # Process points for poly fill
    left_points = np.array([np.transpose(np.vstack([left_fit, point_range]))])
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_fit, point_range])))])
    points = np.hstack((left_points, right_points))
    # Draw lanes as polygons
    cv2.fillPoly(img_color, np.int_(points), (0, 200, 255))
    # Re-warp image
    img_color = perform_image_warp(img_color, dst_size=(frame_width, frame_height), src=ORIG_POINTS, dst=WARP_POINTS)
    # Add original image and colorized lanes
    img_color = cv2.addWeighted(img, 0.5, img_color, 0.7, 0)
    # Return image with colorized lanes
    return img_color