import cv2
from CNN import CNN
from tkinter import messagebox
import numpy as np


class ImageProcessor: #class for handling image processing

    #initializing the image as None, and digits and rectangles as an empty list.
    def __init__(self): 
        self.image = None #will contain the image the user inputs as a cv2 image
        self.digits = [] #will contain the seperate digits in the image
        self.rectangles = [] #will contain the x, y, width and height of the rectangles surrounding any digit, in that order

    #function to load an image from a given path and saves it
    #reshapes it based on the aspect ratio for the UI, so that it doesn't go off screen
    def load_image(self, image_path):
        self.image = cv2.imread(image_path)
        aspect_ratio = self.image.shape[0] / self.image.shape[1]
        self.image = cv2.resize(self.image, (800, 600), fx=aspect_ratio, fy=1/aspect_ratio, interpolation=cv2.INTER_AREA)
        
    #function to grayscale an image and return it
    def to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #function that uses the Gaussian Blur to reduce noise and returns the blurred image
    def reduce_noise(self, image):
        if image is not None:
            return cv2.GaussianBlur(image, (25,25), 0)
        else:
            raise ValueError("No image loaded. Use load_image() first.")
    
    #function to get the image in RGB format
    def get_image(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    #function to resize an image while keeping the aspect ratio, then fill the missing space with padding.
    #arguments: 
    #image - an image as an array
    #target_size - the final size we want the image to be
    #padding color - the color we want to pad. Only one channel, meaning 255 is white and 0 is black
    #returns the resized image.
    def resize_with_padding(self, image, target_size=(28, 28), padding_color=0):

        original_size = image.shape[:2]  #get the original size (height, width)
        target_width, target_height = target_size 
        
        #calculate the aspect ratio
        aspect_ratio = original_size[1] / original_size[0]  # width / height

        #determine the height which we should scale to, based on if width or height are bigger
        if aspect_ratio > 1:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)

        #Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        #Calculate padding to center the image
        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left
        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top

        # Apply padding
        padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padding_color)
        
        return padded_image

    #function to seperate the digits in the picture, as well as proccess them for the neural network. All the previous functions were made for this function 
    #step 1: grayscale the image
    #step 2: reduce noise
    #step 3: binarise it, meaning all pixels will be either black or white, also inverse the colors at the same time
    #THRESH_BINARY_INV is for inversing the colors THRESH_OTSU is for determining the thershold value automatically
    #step 4: find outlines of shapes, or contours
    #RETR_EXTRENAL gets only the outer contours, so no contours inside contours CHAIN_APPROX_SIMPLE is to save time and it only gets the end points of
    #any horizontal, vertical or diagonal line. for example, a rectangle would be represented as 4 points instead of many more.
    #step 5: sort them from left to right (using the x value)
    #step 6: get the smallest rectangle that contains each contour, and save them for later, only if their are is bigger than 150 (to clean noise again)
    #step 7: resize them so that they are 28x28 while keeping their aspect ratio
    def seperate_digits(self):
        if self.image is not None:
            temp = self.image.copy() #so that we don't change the original image
            gray = self.to_grayscale(temp) 
            cv2.imshow("grayscaled", gray)   # These lines are for debugging purposes
            cv2.waitKey(0) 
            blurred = self.reduce_noise(gray)
            cv2.imshow("blurred", blurred)
            cv2.waitKey(0)
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) #adaptive gaussian thresholding
            cv2.imshow("binary inverted", binary)
            cv2.waitKey(0)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #the second value is the hierachy of the contours and is not used
            contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

            self.rectangles = []
            digit = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > 200:
                    self.rectangles.append((x, y, w, h))
                    digit = self.resize_with_padding(binary[y-5:y+h+5, x-5:x+w+5])
                    self.digits.append(digit)
            for i in contours:
                print(i)
            if len(self.digits) == 0:
                messagebox.showerror("Error", "No numbers were found by the algorithm. Try zooming in more?")

        else:
            raise ValueError("No image loaded. Use load_image() first.")
    
    #function to call seperate_digits as well as display what the network recognised as digits to the user
    def show_digits(self):
        if self.image is not None:
            self.seperate_digits()

            image_with_rectangles = self.image.copy()
            for (x, y, w, h) in self.rectangles:
                cv2.rectangle(image_with_rectangles, (x-5, y-5), (x+w+5, y+5+h), (0, 0, 0), 2)
            
            return cv2.cvtColor(image_with_rectangles, cv2.COLOR_BGR2RGB) #convert the image to RGB format for displaying with plt
        else:
            raise messagebox.showerror("Error", "No image loaded. Use load_image() first.")
    
    #function to send all the digits to the neural network to classify them and return the resulting number
    def classify_number(self):
        if self.image is not None:
            if len(self.digits) == 0: #in case we haven't seperated the digits before this
                self.seperate_digits()

            cnn = CNN()
            cnn.load_model()

            final_num = 0
            for i in self.digits:
                final_num *= 10
                final_num += cnn.classify_digit(i)
            
            self.digits = [] #reset self.digits
            
            return final_num
        else:
            raise ValueError("No image loaded. Use load_image() first.")
        


