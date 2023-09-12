import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def detect_car_shadow(original_image, segmentation_mask):
    # Convert the images to grayscale if they are not already.
    if len(original_image.shape) == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    if len(segmentation_mask.shape) == 3:
        segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)

    # Ensure both images have the same size.
    if original_image.shape != segmentation_mask.shape:
        raise ValueError("Original image and segmentation mask must have the same dimensions.")

    # Apply a Gaussian blur to the original image to reduce noise.
    original_image_blur = cv2.GaussianBlur(original_image, (5, 5), 0)

    # Calculate the absolute difference between the original image and its segmentation mask.
    diff = cv2.absdiff(original_image_blur, segmentation_mask)
    # cv2_imshow(diff)
    # Threshold the difference image to obtain a binary mask of potential shadows.
    threshold_value = 20  # Adjust this threshold as needed.
    _, shadow_mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)


    # Use a morphological operation (e.g., dilation) to connect the car with its shadow.
    kernel = np.ones((5, 5), np.uint8)

    shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=1)

    car_shadow_mask = 255-cv2.subtract(shadow_mask, segmentation_mask)

    kernel = np.ones((9,9), np.uint8)  
    closing = cv2.morphologyEx(car_shadow_mask, cv2.MORPH_CLOSE, kernel)
    mask_with_shadow = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    return mask_with_shadow


image_1 = cv2.imread('/content/U-2-Net/dataset/Image/20.jpg')  # Replace 'image_1.jpg' with the actual filename/path of your car image
image_2 = cv2.imread('/content/background.jpg')
image_2 = cv2.resize(image_2, (1920, 1080), interpolation=cv2.INTER_AREA)
car_mask = cv2.imread('/content/U-2-Net/dataset/Mask/20.png', cv2.IMREAD_GRAYSCALE)

car_shadow_mask = detect_car_shadow(image_1, car_mask)

car_mask_bgr = cv2.cvtColor(car_shadow_mask, cv2.COLOR_GRAY2BGR)
# cv2_imshow(car_mask_bgr)

car_mask_gray = cv2.cvtColor(car_mask_bgr, cv2.COLOR_BGR2GRAY)

contours, hierarchy = cv2.findContours(car_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

c = max(contours, key=cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)

car = cv2.bitwise_and(image_1, car_mask_bgr)
# cv2_imshow(car)
# Extract car image using bounding box
car_img = car[y:y+h, x:x+w]
car_mask_a = car_mask_bgr[y:y+h, x:x+w]
# cv2_imshow(car_img)
# cv2_imshow(car_mask_a)

x_position , y_position = 0,image_2.shape[0]-car_img.shape[0]

# new_width , new_height = int(car_img.shape[1]*0.70), int(car_img.shape[0]*0.70)
ratio = car_img.shape[0]/car_img.shape[1]
new_width  = int(image_2.shape[1]*0.70)
new_height = int(new_width*ratio)

resized_car = cv2.resize(car_img, (new_width, new_height))
resized_mask = cv2.resize(car_mask_a, (new_width, new_height))

x_position = (image_2.shape[1] - new_width)//2
y_position = image_2.shape[0]-resized_car.shape[0]-int(image_2.shape[0]*0.05)

new_x, new_y = (x_position, y_position)

result = np.zeros_like(image_2)
result_mask = np.zeros_like(image_2)

result[new_y:new_y + new_height, new_x:new_x + new_width] = resized_car
result_mask[new_y:new_y + new_height, new_x:new_x + new_width] = resized_mask

background_mask = 255 - result_mask
background = cv2.bitwise_and(image_2, background_mask)
car_virtual = cv2.add(result, background)
cv2_imshow(car_virtual)