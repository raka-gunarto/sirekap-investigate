import cv2
import numpy as np
import logging
import argparse
import sys
import os
import json
from tqdm.contrib.concurrent import thread_map, process_map

EXPECTED_STATIONS = 823236

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    LOG_LEVELS = {
        logging.DEBUG: (grey, ""),
        logging.INFO: (blue, ""),
        logging.WARNING: (yellow, ""),
        logging.ERROR: (red, ""),
        logging.CRITICAL: (bold_red, ""),
    }

    def format(self, record):
        log_fmt = self.LOG_LEVELS.get(record.levelno, (self.grey, ""))
        color, emoji = log_fmt
        formatter = logging.Formatter(f"%(asctime)s - [{color}%(levelname)s{self.reset}{emoji}]: %(message)s")
        return formatter.format(record)

def setup_logger(log_level, filename):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(CustomFormatter())

    fh = logging.FileHandler(filename, mode="w")
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s]: %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def parse_args():
    args = argparse.ArgumentParser(description="Sirekap data OMR processor")
    args.add_argument("-l", "--log-level", default="INFO", help="Log level")
    args.add_argument("-f", "--log-file", default="sirekap-data-omr.log", help="Log file")
    args.add_argument("-i", "--images-dir", required=True, help="Directory containing images to process")
    args.add_argument("-p", "--polling-stations", required=True, help="Polling station cache file")
    args.add_argument("-c", "--concurrency", default=32, type=int, help="Number of processes/threads to use")
    args.add_argument("--limit", type=int, help="Limit the number of images to process")

    return args.parse_args()

params = cv2.aruco.DetectorParameters()
params.aprilTagQuadDecimate = 1.0
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H10)
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

def crop_and_correct(gray):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids == None or len(ids[0]) != 1:
        return (None, None, "Apriltags detected != 1")
    tag_corners_0 = np.array(corners[0][0], dtype="float32")
    tag_corners = tag_corners_0[[2, 3, 0, 1], :]

    # Calculate rotation angle
    dx = tag_corners[1][0] - tag_corners[0][0]
    dy = tag_corners[1][1] - tag_corners[0][1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    # Center of the tag for rotation
    tag_center = np.mean(tag_corners, axis=0)

    # Rotation matrix
    M = cv2.getRotationMatrix2D((tag_center[0], tag_center[1]), angle_deg, 1)

    # Size of the original image
    h, w = gray.shape[:2]

    # Create a large canvas to avoid clipping issues
    canvas_size = (5 * w, 5 * h)

    # Compute the translation to center the original image on the large canvas
    tx = (canvas_size[0] - w) // 2
    ty = (canvas_size[1] - h) // 2

    # Adjust the rotation matrix to include the translation
    M[0, 2] += tx - tag_center[0]
    M[1, 2] += ty - tag_center[1]

    # Apply the rotational correction
    corrected_img = cv2.warpAffine(gray, M, canvas_size)

    # Transform the tag corners using the rotation matrix
    tag_corners = np.float32([tag_corners])
    transformed_tag_corners = cv2.transform(tag_corners, M)[0]

    # Assume we're using the left-most corners (top-left and bottom-left) for simplicity
    # Adjust indices as per your tag's corner order
    bottom_left_corner, top_left_corner = transformed_tag_corners[3], transformed_tag_corners[0]
    vertical_skew_dy = bottom_left_corner[1] - top_left_corner[1]
    vertical_skew_dx = bottom_left_corner[0] - top_left_corner[0]

    vertical_skew_angle_rad = np.arctan2(vertical_skew_dx, vertical_skew_dy)  # Angle relative to horizontal
    vertical_skew_angle_deg = np.degrees(vertical_skew_angle_rad)

    # Check for excessive skew angle which may indicate an incorrect calculation
    if abs(vertical_skew_angle_deg) > 40:
        return (None, None, "Too Much Vertical Skew")
    else:
        # Vertical skew correction matrix, shearing based on x-shifts for vertical alignment
        vertical_skew_correction_matrix = np.float32([
            [1, 0, 0],
            [np.tan(vertical_skew_angle_rad), 1, 0]
        ])

        # Apply vertical skew correction
        corrected_img = cv2.warpAffine(corrected_img, vertical_skew_correction_matrix, canvas_size)

        # Transform the tag corners using the vertical skew correction matrix
        transformed_tag_corners = cv2.transform(np.array([transformed_tag_corners]), vertical_skew_correction_matrix)[0]

    top_left_corner, top_right_corner = transformed_tag_corners[0], transformed_tag_corners[1]
    horizontal_skew_dy = top_right_corner[1] - top_left_corner[1]
    horizontal_skew_dx = top_right_corner[0] - top_left_corner[0]

    # Calculate the angle of skew relative to vertical
    horizontal_skew_angle_rad = np.arctan2(horizontal_skew_dy, horizontal_skew_dx)
    horizontal_skew_angle_deg = np.degrees(horizontal_skew_angle_rad)

    # Check for excessive skew angle which may indicate an incorrect calculation
    if abs(horizontal_skew_angle_deg) > 40:
        return (None, None, "Too Much Horizontal Skew")
    else:
        # Horizontal skew correction matrix, shearing based on y-shifts for horizontal alignment
        horizontal_skew_correction_matrix = np.float32([
            [1, -np.tan(horizontal_skew_angle_rad), 0],
            [0, 1, 0]
        ])

        # Apply horizontal skew correction
        # Assuming `corrected_img` is the image after vertical skew correction
        corrected_img = cv2.warpAffine(corrected_img, horizontal_skew_correction_matrix, (corrected_img.shape[1], corrected_img.shape[0]))

        # Transform the tag corners using the horizontal skew correction matrix
        transformed_tag_corners = cv2.transform(np.array([transformed_tag_corners]), horizontal_skew_correction_matrix)[0]

    # Compute new boundaries for cropping based on the rotated tag
    new_min_x, new_min_y = np.min(transformed_tag_corners, axis=0).astype(int)
    new_max_x, new_max_y = np.max(transformed_tag_corners, axis=0).astype(int)

    # Calculate the new width and height based on the transformed tag
    new_tag_width = new_max_x - new_min_x
    new_max_x = int(new_max_x + new_tag_width * 0.5)  # Adjust as per original logic
    new_tag_width = new_max_x - new_min_x
    new_tag_height = new_max_y - new_min_y
    new_min_y = int(new_min_y - new_tag_height * 0.2)  # Adjust as per original logic
    new_tag_height = new_max_y - new_min_y

    # Calculate new dimensions for cropping based on the specified multipliers
    new_crop_width = int(new_tag_width * 2.3)
    new_crop_height = int(new_tag_height * 15)

    # Adjust the cropping area
    crop_top_left_x = new_max_x - new_crop_width
    crop_top_left_y = new_min_y  # Assuming the tag is at the top

    # Ensure the crop area is within the bounds of the new canvas
    crop_top_left_x = max(0, crop_top_left_x)
    crop_top_left_y = max(0, crop_top_left_y)
    crop_bottom_right_x = min(crop_top_left_x + new_crop_width, canvas_size[0])
    crop_bottom_right_y = min(crop_top_left_y + new_crop_height, canvas_size[1])

    # Crop the image using the adjusted coordinates
    cropped_img_with_tag = corrected_img[crop_top_left_y:crop_bottom_right_y, crop_top_left_x:crop_bottom_right_x]
    return (cropped_img_with_tag, ids[0][0], None)

def circle_detect(cropped):
    resized = cv2.resize(cropped, (250, 1500), interpolation=cv2.INTER_CUBIC)

    thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 1)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_radius = 8  # Adjust based on your image
    max_radius = 12 # 240  # Adjust based on your image
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        # Calculate area of the minimum enclosing circle
        if min_radius < radius < max_radius:
            circle_area = np.pi * (radius ** 2)

            # Compare the contour area to the circle area, allow some margin of error
            # The smaller the difference, the more likely it is to be a circle
            if abs(area - circle_area) / circle_area < 0.15:  # You can adjust the threshold
                circles.append(cnt)

    def non_max_suppression_contours(contours, overlapThresh):
        # Check if there are any contours
        if len(contours) == 0:
            return []

        # Initialize the list of bounding boxes and the list of corresponding scores (area in this case)
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)  # Calculate bounding box for each contour
            boxes.append([x, y, w, h])
        boxes = np.array(boxes).astype("float")

        # Initialize the list of picked indexes
        pick = []

        # Grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = x1 + boxes[:,2]
        y2 = y1 + boxes[:,3]

        # Compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1) * (y2 - y1)
        idxs = np.argsort(y2)

        # Keep looping while some indexes still remain in the list
        while len(idxs) > 0:
            # Grab the last index in the indexes list, add the index value to the list of picked indexes, then initialize the suppression list (i.e., indexes that will be deleted) starting with the last index
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]

            # Loop over all indexes in the indexes list
            for pos in range(0, last):
                # Grab the current index
                j = idxs[pos]

                # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])

                # Compute the width and height of the bounding box
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)

                # Compute the ratio of overlap between the computed bounding box and the bounding box in the area list
                overlap = (w * h) / area[j]

                # If there is sufficient overlap, suppress the current bounding box
                if overlap > overlapThresh:
                    suppress.append(pos)

            # Delete all indexes from the index list that are in the suppression list
            idxs = np.delete(idxs, suppress)

        # Return the contours that were picked using the index list
        return [contours[i] for i in pick]

    circles = non_max_suppression_contours(circles, 0.3)
    return circles, thresh

def circles_process_grid(circles, thresh):
    processed_circles = []
    for cnt in circles:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        # Create a blank mask with the same dimensions as the thresholded image
        mask = np.zeros(thresh.shape, np.uint8)
        
        # Draw the filled circle on the mask
        cv2.circle(mask, center, radius, 255, -1)
        
        # Apply the mask to the thresholded image
        contour_area = cv2.bitwise_and(thresh, thresh, mask=mask)
        
        # Count the number of white pixels within the circle
        white_pixels = np.sum(contour_area == 255)
        
        # Calculate the filled ratio within the circle
        circle_area = np.pi * (radius ** 2)
        filled = white_pixels / circle_area > 0.75  # Use the area of the circle for the filled ratio
        
        # Append the circle information including its filled status
        processed_circles.append((center[0], center[1], radius, filled, cnt))

    processed_circles.sort(key=lambda x: x[1])
    rows = [[]]
    current_row = 0

    for i, circle in enumerate(processed_circles):
        x, y, radius, filled, cnt = circle

        if i == 0 or abs(y - processed_circles[i-1][1]) <= radius * 0.5:
            rows[current_row].append(circle)
        else:
            current_row += 1
            rows.append([circle])

    for row in rows:
        row.sort(key=lambda x: x[0])
    
    return rows

def get_votes(circle_rows, expected_rows, expected_columns):
    if len(circle_rows) != expected_rows: return (None, "Incorrect no. of rows")
    if any([len(row) != expected_columns for row in circle_rows]): return (None, "Incorrect no. of columns")

    def index_with_default(iter, val, default=0):
        try:
            return iter.index(val)
        except:
            return default

    candidate1 = ''.join([str(index_with_default(col, True)) for col in [[row[i][3] for row in circle_rows[0:10]] for i in range(expected_columns)]])
    candidate2 = ''.join([str(index_with_default(col, True)) for col in [[row[i][3] for row in circle_rows[10:20]] for i in range(expected_columns)]])
    candidate3 = ''.join([str(index_with_default(col, True)) for col in [[row[i][3] for row in circle_rows[20:30]] for i in range(expected_columns)]])

    return ([candidate1, candidate2, candidate3], None)

def process_image(image):
    station, image_path = image
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cropped_img_with_tag, id, reason = crop_and_correct(image)
        if cropped_img_with_tag is None:
            return (station, None, reason)
        
        circles, thresh = circle_detect(cropped_img_with_tag)
        circle_rows = circles_process_grid(circles, thresh)

        expected_rows = 30
        expected_columns = 3 if id == 102 else 5
        votes, reason = get_votes(circle_rows, expected_rows, expected_columns)

        if votes is None:
            return (station, None, reason)
        
        return (station, votes, None)
        

    except Exception as e: return (station, None, f"Exception occured: {str(e)}")

def process_images(image_dir, stations, workers, limit = None):
    # construct input list
    inputlist = []
    for station, data in stations.items():
        url = data['images'][1]
        if not url: continue

        inputlist.append((station, os.path.join(image_dir, url.split("/")[-1])))
    
    if limit: inputlist = inputlist[:limit]

    # process images in parallel
    logging.info(f"Processing {len(inputlist)} images using {workers} workers")
    results = process_map(process_image, inputlist, max_workers=workers, chunksize=100)

    with open("results.json", 'w') as f:
        output = {}
        for station, result, fail_reason in results:
            output[station] = {
                'result': result,
                'fail_reason': fail_reason,
            }
        json.dump(output, f)


def main():
    args = parse_args()
    setup_logger(args.log_level, args.log_file)

    logging.info("Starting Sirekap Data OMR Processor...")
    with open(args.polling_stations, "r") as f:
        stations = json.load(f)

    process_images(args.images_dir, stations, args.concurrency, args.limit)

if __name__ == "__main__":
    main()
