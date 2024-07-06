import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
import matplotlib.pyplot as plt


class CNN: #class to control the cnn network and related things

    #function to get the data from mnist, normalise it, and return it.
    @staticmethod
    def load_training_data():

        #loading the data from mnist digits dataset
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        #normalizing the data to fit the shape required, also change 0-255 for color into 0-1
        x_train = x_train.reshape(-1,28,28,1)
        x_train = x_train/255.0 

        x_test = x_test.reshape(-1,28,28,1)
        x_test = x_test/255.0

        #One hot encoding; for example, change 7 into [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test)

    def __init__(self):
        #getting training data from the data loader
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_training_data()
        self.model = None
    
    #function to create a new model
    def create_model(self):
        model = tf.keras.models.Sequential() #this is the model
        model.add(Conv2D(32, kernel_size=(3,3), kernel_initializer='he_uniform', activation="relu", input_shape=(28,28,1)))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(25, activation="relu"))
        model.add(Dense(10, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        self.model = model #saving the model

    #function to train the model
    def train_model(self, epochs):
        history = self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=epochs, validation_data=(self.x_test, self.y_test))
        self.plot_history(history)

    #function to plot the models performance on a graph
    def plot_history(self, history):
        #create two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        #plot accuracy on the first subplot
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_ylabel("Accuracy")
        ax1.set_xlabel("Epochs")
        ax1.legend(loc='lower right')
        ax1.set_title('Accuracy Evaluation')
        
        #plot loss on the second subplot
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_ylabel("Loss")
        ax2.set_xlabel("Epochs")
        ax2.legend(loc='upper right')
        ax2.set_title('Loss Evaluation')
        
        #adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    #function to evaluate the model
    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=2) #loss is the loss of the network and accuracy is the accuracy in it's classification (*100 for percentage)
        print(f"Accuracy: {accuracy}")
        print(f"Loss: {loss}")
    
    #function to load a built model
    def load_model(self):
        self.model = tf.keras.models.load_model("DigitClassifier.keras")
    
    #function to classify a specific digit, returns the models prediction for that specific number
    def classify_digit(self, image):
        if self.model is not None:
            return np.argmax(self.model.predict(image.reshape(1,28,28,1)))
        else:
            print("the model wasn't loaded, please load it before trying to use it!")
    
    def save_new_model(self):
        if self.model is not None:
            self.model.model.save("DigitClassifier.keras")
        else:
            raise ValueError("No model loaded or created")
    
    
if __name__ == "__main__":
    cnn = CNN()
    cnn.create_model()
    cnn.train_model(10)
    print(cnn.model.summary())
