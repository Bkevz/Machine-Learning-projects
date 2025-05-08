# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\opencv\ui\Gujarathi_lang_recognition\demo2.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

# GUI imports
from PyQt5 import QtCore, QtGui, QtWidgets

# Data processing imports
import numpy as np
import cv2
import os
import csv
import imutils
import mahotas as mt

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Sequential, model_from_json
    from tensorflow.keras.layers import (
        Dense, 
        Conv2D, 
        MaxPooling2D, 
        Flatten, 
        BatchNormalization,
        Dropout
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Please ensure TensorFlow is properly installed")
    raise

# Machine learning imports
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from sklearn.neural_network import MLPClassifier

# Scientific computing
import scipy
import scipy.io as sio

p = 1

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(160, 370, 151, 51))
        self.BrowseImage.setObjectName("BrowseImage")
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(200, 80, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 20, 621, 20))
        font = QtGui.QFont()
        font.setFamily("Courier New")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160, 450, 151, 51))
        self.Classify.setObjectName("Classify")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(430, 370, 111, 16))
        self.label.setObjectName("label")
        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(400, 450, 151, 51))
        self.Training.setObjectName("Training")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(400, 390, 211, 51))
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.BrowseImage.clicked.connect(self.loadImage)

        self.Classify.clicked.connect(self.classifyFunction)

        self.Training.clicked.connect(self.trainingFunction)        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BrowseImage.setText(_translate("MainWindow", "Browse Image"))
        self.label_2.setText(_translate("MainWindow", "            COVID-19 DETECTION"))
        self.Classify.setText(_translate("MainWindow", "Classify"))
        self.label.setText(_translate("MainWindow", "Recognized Class"))
        self.Training.setText(_translate("MainWindow", "Training"))

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") # Ask for file
        if fileName: # If the user gives a file
            print(fileName)
            self.file=fileName
            pixmap = QtGui.QPixmap(fileName) # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
            self.imageLbl.setPixmap(pixmap) # Set the pixmap onto the label
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center

    def classifyFunction(self):
        try:
            import tensorflow as tf
            
            # Load the model with .keras extension
            loaded_model = tf.keras.models.load_model('complete_model.keras')
            
            label = ["Covid", "Normal"]
            path2 = self.file
            print(path2)
            
            # Process the image
            test_image = tf.keras.preprocessing.image.load_img(path2, target_size=(128, 128))
            test_image = tf.keras.preprocessing.image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            
            # Normalize the image
            test_image = test_image / 255.0
            
            # Make prediction
            result = loaded_model.predict(test_image)
            print(result)
            label2 = label[result.argmax()]
            print(label2)
            self.textEdit.setText(label2)
            
        except Exception as e:
            print(f"Error loading/predicting with model: {str(e)}")
            self.textEdit.setText(f"Error: {str(e)}")

    def trainingFunction(self):
        try:
            self.textEdit.setText("Training under process...")
            import tensorflow as tf
            
            # Create the model
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(96, kernel_size=(3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(2, activation='softmax')
            ])

            model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

            test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

            training_path = './TrainingDataset'
            testing_path = './TestingDataset'

            training_set = train_datagen.flow_from_directory(
                training_path,
                target_size=(128, 128),
                batch_size=8,
                class_mode='categorical')
            
            test_set = test_datagen.flow_from_directory(
                testing_path,
                target_size=(128, 128),
                batch_size=8,
                class_mode='categorical')

            model.fit(
                training_set,
                steps_per_epoch=100,
                epochs=10,
                validation_data=test_set,
                validation_steps=125)

            # Save the model with .keras extension
            model.save('complete_model.keras')
            
            print("Saved model to disk")
            self.textEdit.setText("Training completed and model saved")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            self.textEdit.setText(f"Training error: {str(e)}")
        
        
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

