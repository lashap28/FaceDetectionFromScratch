from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix import textinput
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from collections import Counter
import cv2
import tensorflow as tf
import FuncWeights as fw
from random import sample
import os
import uuid
import numpy as np
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


class CamAppTwo(App):

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
        self.text = Label(text='Checking In The Base:', size_hint=(1, .1))
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.text)
        self.LoadModel()
        for f in os.listdir('Find_Face'):
            os.remove(os.path.join('Find_Face', f))

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        self.event = Clock.schedule_interval(self.WhoIsIt, 0.5)
        return layout

    def update(self, *args):

        ret, frame = self.capture.read()
        pixels = np.asarray(frame)
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(r'../FaceIdentify/haarcascade_frontalface_default.xml')
        results = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(results) != 0:
            (x, y, w, h) = results[0]
            w = int(np.ceil(w * 0.85))
            h = int(np.ceil(h * 1.18))
            y = int(np.ceil(0.8 * y))
            x = int(np.ceil(x * 1.11))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (96, 96))
            if not os.path.exists('Find_Face'):
                os.makedirs('Find_Face')
            cv2.imwrite(os.path.join('Find_Face', f'pic_{uuid.uuid1()}.jpg'), face)
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def WhoIsIt(self, *args):

        if len(os.listdir('Find_Face')) > 10:

            distance = []
            mind = 100
            pname = ''
            for pic in os.listdir('Find_Face'):
                perc = cv2.imread(os.path.join('Find_Face', pic))
                face_cc = fw.RImgToEncoding(perc, self.AppModel)

                for p in os.listdir('Embed_Face'):
                    for pp in os.listdir(os.path.join('Embed_Face', p)):
                        f = np.genfromtxt(os.path.join('Embed_Face', p, pp), delimiter=',', dtype=None)
                        distance.append(np.linalg.norm(face_cc - f))

                    if min(distance) < mind:
                        mind = min(distance)
                        pname = p
            self.text.text = pname + ' With Distance: ' + str(np.round(mind,2))
            self.event.cancel()
            return mind, pname


if __name__ == '__main__':
    CamAppTwo().run()
