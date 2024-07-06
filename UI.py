import tkinter as tk
from ImageDataLoader import ImageProcessor
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

#class for the app, controls the UI and 
class NumberRecognitionApp(tk.Tk):

    BUTTON_FONT = ("Ariel", 20) #button font

    #init function, which would initialise the window and the widjects in it
    def __init__(self): 
        super().__init__() #get all the widjets from Tk so we don't have to worry about it
        self.title("Number Recognition App") #title
        self.geometry("1200x900") #size of the window
        self.configure(bg="gray") #set background to gray

        self.processor = ImageProcessor() #initialise the image proccesor
        self.is_image = False #variable to remember if we uploaded an image

        #making the button to upload images
        self.upload_button = tk.Button(self, text="Upload image", font=self.BUTTON_FONT, command=self.upload_image)
        self.upload_button.pack(side="top", fill="x", padx=20, pady=20)

        #label to show the image
        self.image_label = tk.Label(self, text="Image will go here", font=self.BUTTON_FONT)
        self.image_label.pack(pady=20, padx=20, fill="both", anchor="center", expand=True)

        #label to show results
        self.result_label = tk.Label(self, text="Results:", justify="center", anchor="center", font=self.BUTTON_FONT)
        self.result_label.pack(fill="both")

        #making the button to analyse image
        self.results_button = tk.Button(self, text="Analyse image", font=self.BUTTON_FONT, command=self.show_results)
        self.results_button.pack(side="bottom", fill='x', padx=20, pady=20)

    #function to show an image that is in cv2 form (because the processor saves it in cv2)
    #takes an image as an array in RGB format
    def show_image_from_array(self, image_as_array):
        image_pil = Image.fromarray(image_as_array) #getting the image as a PIL image
        image_tk = ImageTk.PhotoImage(image_pil) #getting the image as a photoimage for TKinter
        self.image_label.configure(image=image_tk) #setting the label to the image
        self.image_label.image = image_tk #saving the image

    #function to upload a image and load it into the image processor
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.processor.load_image(file_path)
                self.show_image_from_array(self.processor.get_image())
                self.is_image = True
                messagebox.showinfo("Success", "Image uploaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}\nThe file is not in a compatible image format.")

    #function that:
    #shows the image with rectangles surrounding the digits the network detected
    #displays the number the network classified on screen
    def show_results(self):
        if self.is_image:
            try:
                self.show_image_from_array(self.processor.show_digits())
                self.result_label.config(text=f"The number classified by the network is: {self.processor.classify_number()}")
            except Exception as e:
                print(e)
                messagebox.showerror("Error", "There was an unexpected error: \n" + str(e))
                self.result_label.config(text="Error")
        else:
            messagebox.showerror("Error", "Please upload image first!")
