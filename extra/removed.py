#Dont think these are necessary
### Preprocessing helpers
'''
def preprocess_image(image):
    image_cropped = image[100:440, :-90] # -> (380, 550, 3)
    image = cv2.resize(image_cropped, (220, 66), interpolation = cv2.INTER_AREA)
    return image


def preprocess_image_valid_from_path(image_path, speed):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img)
    return img, speed
'''

