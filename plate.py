import torch
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
pathlib.PosixPath = pathlib.WindowsPath 
model = torch.hub.load('ultralytics/yolov5', 'custom', path='src/model.pt')


def blur_plate(boxes, img):
    for box in boxes:
        x1, y1, x2, y2, confidence, cls = box.astype(int)  # Convert to integers
        roi = img[y1:y2, x1:x2]
        roi_blurred = cv2.GaussianBlur(roi, (33, 33), 0)  
        img[y1:y2, x1:x2] = roi_blurred


def ndarray_to_boolean_mask(ndarray, threshold=8):
  mask = ndarray>=threshold 
  return mask

def adjust_array(arr):
    rows, cols = arr.shape
    thr = 50
    for i in range(rows):
        for j in range(1, cols - 1):  # Skip the first and last elements of each row
            if arr[i, j] < thr and (arr[i, j - 1] > thr or arr[i, j + 1] > thr):
                arr[i, j] = 255
    cv2.imshow("result",arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return arr

def sobel_edge_detection_in_box(image, boxes):
    for box in boxes:
        x1, y1, x2, y2, confidence, cls = box.astype(int)  # Convert to integers
        roi = image[y1:y2,x1:x2]
        
        # Convert ROI to grayscale if it's not already
        if len(roi.shape) == 3:  
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        adjust_array(roi)
        # Define Sobel kernels
        sobel_x = np.array([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1], 
                            [0, 0, 0], 
                            [1, 2, 1]])

        # Apply Sobel convolution
        grad_x = cv2.filter2D(roi, cv2.CV_64F, sobel_x)
        grad_y = cv2.filter2D(roi, cv2.CV_64F, sobel_y)

        # Calculate the gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # print(grad_magnitude[0,:20])
        # Normalize for display
        grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # cv2.imshow("result",grad_magnitude)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Merge the modified ROI back into the original image
        grad_magnitude = ndarray_to_boolean_mask(grad_magnitude)
        image[y1:y2,x1:x2][grad_magnitude] = (255,255,255) 
        

    return image


image_path = 'car3.png'

# Run inference
results = model(image_path)
boxes = results.xyxy[0].cpu().numpy() 
img = cv2.imread(image_path)


# img = sobel_edge_detection_in_box(img, boxes)
blur_plate(boxes, img)


# Display the resulting image without bounding boxes
cv2.imshow("result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

