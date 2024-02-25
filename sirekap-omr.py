# work in progress

import cv2
import numpy as np
import concurrent.futures
from tqdm import tqdm

images = open('images.txt', 'r')
images = ["/mnt/data-sirekap/" + x.strip() for x in images]

def process_image(image):
    try:
        form = cv2.imread(image)
        if form is None:  # Check if the image was not loaded properly
            return False
        gray = cv2.cvtColor(form, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] > 500 and len(cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)) == 4]
        result = len(contours) == 2
    except Exception as e:
        print(f"Error processing image {image}: {e}")
        result = False

    return result, image

def process_images(images):
    valid = 0
    good_images = []
    bad_images = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        progress = tqdm(total=len(images), desc="Processing Images")  # Initialize the progress bar
        # Submit all the tasks and create futures
        futures = [executor.submit(process_image, image) for image in images]
        # As futures complete, results will be yielded
        for future in concurrent.futures.as_completed(futures):
            result, image = future.result()
            if result:
                valid += 1
                good_images.append(image)
            else: bad_images.append(image)
            progress.update(1)
        progress.close()  # Close the progress bar when done
    return valid, good_images, bad_images

valid, good_images, bad_images = process_images(images[0:1000])
print(f"{valid} good images out of {len(images)}")

with open('good_images.txt', 'w') as f:
    for image in good_images:
        f.write(image + '\n')

with open('bad_images.txt', 'w') as f:
    for image in bad_images:
        f.write(image + '\n')