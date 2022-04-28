from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix import textinput
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from numpy import savetxt
from kivy.logger import Logger
from PIL import Image as Im
import cv2
import tensorflow as tf
import FuncWeights as fw
import InceptionBlocks as IB
import os
import uuid
from mtcnn.mtcnn import MTCNN
import numpy as np


class CamApp(App):

    def TripletLoss(self, y_true, y_pred, alpha=0.2):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

        basic_loss = pos_dist - neg_dist + alpha
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

        return loss

    def LoadModel(self):
        self.AppModel = tf.keras.models.load_model(r'App_Model/', custom_objects={'TripletLoss': self.TripletLoss})

    def build(self):

        self.web_cam = Image(size_hint=(1, .8))
        self.text = Label(text='Input Your Name For Adding In The DataBase', size_hint=(1, .1))
        self.inptext = textinput.TextInput(text='Bob Dylan', size_hint=(1, .1), multiline=False)
        self.button = Button(text='Capture', on_press=self.rawIm, size_hint=(1, .1))
        self.buttonF = Button(text=f'Embedding', disabled=True, on_press=self.FaceDetect, size_hint=(1, .1))


        # add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.text)
        layout.add_widget(self.inptext)
        layout.add_widget(self.button)
        layout.add_widget(self.buttonF)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return layout

    def update(self, *args):
        ret, frame = self.capture.read()
        pixels = np.asarray(frame)
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(r'../FaceIdentify/haarcascade_frontalface_default.xml')
        results = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in results:
            w = int(np.ceil(w * 0.85))
            h = int(np.ceil(h * 1.18))
            y = int(np.ceil(0.8 * y))
            x = int(np.ceil(x * 1.11))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture
        if os.path.exists(os.path.join('Face_Library_Raw', self.inptext.text)):
            if len(os.listdir('Face_Library_Raw' + f'/{self.inptext.text}')) > 5:
                self.buttonF.disabled = False

    def rawIm(self, *args):

        SAVE_RAW = 'Face_Library_Raw'
        ret, frame = self.capture.read()
        frame = cv2.resize(frame, (96, 96))

        if not os.path.exists(os.path.join(SAVE_RAW, self.inptext.text)):
            os.makedirs(os.path.join(SAVE_RAW, self.inptext.text))
        cv2.imwrite(os.path.join(SAVE_RAW, self.inptext.text, self.inptext.text + '_' + str(uuid.uuid1()) + '.jpg'),
                    frame)

    def FaceDetect(self, *args):

        self.LoadModel()

        for person in os.listdir('Face_Library_Raw'):
            for person_face in os.listdir('Face_Library_Raw' + f'/{person}'):

                image = cv2.imread('Face_Library_Raw' + f'/{person}' + '/' + person_face)
                pixels = np.asarray(image)
                gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(r'../FaceIdentify/haarcascade_frontalface_default.xml')
                results = face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(results) > 0:

                    x1, y1, width, height = results[0]
                    width = int(np.ceil(width * 0.85))
                    height = int(np.ceil(height * 1.18))
                    y1 = int(np.ceil(0.8 * y1))
                    x1 = int(np.ceil(x1 * 1.11))
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height

                    face = pixels[y1:y2, x1:x2]
                    face = cv2.resize(face, (96, 96))
                    image = Im.fromarray(face)
                    uid = str(uuid.uuid1())
                    if not os.path.exists(os.path.join('Embed_Face', person)):
                        os.makedirs('Embed_Face' + '/' + person)
                    path1 = os.path.join('Embed_Face', person)
                    savetxt(path1 + '//' + f'{person}+_embed+_{uid}.csv', fw.RImgToEncoding(face, self.AppModel),
                            delimiter=',')

                    if not os.path.exists(os.path.join('Only_Face', person)):
                        os.makedirs('Only_Face' + '/' + person)
                    path2 = os.path.join('Only_Face', person)
                    image.save(f"Only_Face//{person}//person_{uid}.jpg")

if __name__ == '__main__':
    CamApp().run()
