# Import dependencies
import cv2, pickle
import numpy as np

picklefile = "train_data.p"
# train_labels_filename = "train_labels.p"

# Import images from pickle file
image_array = np.array(pickle.load(open(picklefile, 'rb')))

# Iterate over image array
for cnt in range(len(image_array)):
    # Log
    print("Image", cnt, "of", len(image_array))
    filename = 'images_1/frame_{0:0>4}.png'.format(cnt)
    cv2.imwrite(filename, image_array[cnt])

# Log
print("Done.")