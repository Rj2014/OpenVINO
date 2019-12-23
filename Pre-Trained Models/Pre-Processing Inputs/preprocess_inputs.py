import cv2
import numpy as np

def preprocessing(input_image, height, width):
    '''
    The images coming from cv2.imread were already going to be BGR, 
    and all the models wanted BGR inputs. However, each image was 
    coming in as height x width x channels, and each of these networks
    wanted channels first, along with an extra dimension at the start 
    for batch size. 
    So, for each network, and Given an input image, height and width,
    the preprocessing needed to:
    - Resize to height and width
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start  
    '''
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image

def pose_estimation(input_image):
    preprocessed_image = np.copy(input_image)
    preprocessed_image = preprocessing(preprocessed_image, 256, 456)
    return preprocessed_image


def text_detection(input_image):
    
    preprocessed_image = np.copy(input_image)
    preprocessed_image = preprocessing(preprocessed_image, 768, 1280)
    return preprocessed_image


def car_meta(input_image):
   
    preprocessed_image = np.copy(input_image)
    preprocessed_image = preprocessing(preprocessed_image, 72, 72)
    return preprocessed_image
