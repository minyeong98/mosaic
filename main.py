from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import cv2
import numpy as np
import face_recognition
from PIL import Image

import sys
import os
import warnings
import sqlite3
import io

name = "Human Mosaic"

class humic:
    def __init__(self, cursor, con):
        self.known_encodings, self.known_names = [], []
        self.cursor, self.con = cursor, con
        self.process_this_frame = True

    def call_face(self):
        # database select query
        m = self.cursor.execute("""SELECT * FROM FACE""") #sql 쿼리실행
        self.con.commit()
        # DB 경로의 이미지 파일을 불러와 미리 학습
        for x in m: #저장되어 있는 이미지를 불러서 encoding 후 128개의 특징점 값 조절
            name, ext = os.path.splitext(x[0])
            if ext == ".jpg":
                print(name)
                try:
                    self.known_names.append(name)
                    img = np.array(Image.open(io.BytesIO(x[1])))
                    face_encoding = face_recognition.face_encodings(img)[0] #128개의 특징점을 다른 사람의 특징점과 비교
                    self.known_encodings.append(face_encoding)
                except:
                    print("exception")

        self.locations = []
        self.face_encodings = []
        self.face_names = []

    def __del__(self):
        pass
    #얼굴인식
    def get_frame(self, frame, mosaic):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
    #얼굴검출
        if self.process_this_frame: #face_recognition 라이브러리에 있는 face locations & encodings를 이용하여 얼굴 검출 및 저장
            self.locations = face_recognition.face_locations(rgb_small_frame) #얼굴 좌표값 저장
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.locations) #인코딩하여 128개의 값으로 변환

            self.face_names = []
            self.face_dist = []
            for face_encoding in self.face_encodings: #face_encoding 함수
                distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                min_value = min(distances)

                name = "WHO"

                if min_value < 0.45:  #검출된 encoding 값 vs known_encoding 값
                    idx = np.argmin(distances)
                    name = self.known_names[idx]

                self.face_names.append(name)
                self.face_dist.append(sum(distances) / len(distances))

        self.process_this_frame = not self.process_this_frame
        #화면표시
        for (top, right, bottom, left), name, dist in zip(self.locations, self.face_names, self.face_dist):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            font = cv2.FONT_HERSHEY_SIMPLEX

            if name == "WHO":
                # 모자이크
                if mosaic:
                    face_img = frame[top:bottom, left:right]
                    face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.10, fy=0.10)
                    face_img = cv2.resize(face_img, (right - left, bottom - top), interpolation=cv2.INTER_AREA) #영역적인 정보를 추출해서 결과영상을 보여줌
                    frame[top:bottom, left:right] = face_img
                #화면상 얼굴의 사각표시
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                #화면상 얼굴의 사각표시 밑에 표시할 문자 위치
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                #사각 표시할 문자 설정
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            # 학습된 얼굴
            else:
                #영상에 박스 및 문자생성
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, "%s(%d%s)" % (name, int(dist * 100), "%"), (left + 6, bottom - 6), font, 0.75, (255, 255, 255), 1)

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        _, jpg = cv2.imencode(".jpg", frame)
        return jpg.tobytes()


# 쓰레드 작업 1 (동시에 동작을 수행하기 위한 작업)
class ShowVideo(QObject):

    flag, mosaic_flag = 0, False
    VideoSignal = pyqtSignal(QImage)
    camera = cv2.VideoCapture(cv2.CAP_DSHOW) #해상도 1080


    def __init__(self, cursor, con):
        super(ShowVideo, self).__init__(parent=None)
        self.face_recog = humic(cursor, con)
        self.run_video = False
    # start 버튼 클릭 시
    @pyqtSlot()
    def startVideo(self):
        global image

        while self.run_video:
            _, image = self.camera.read()
            height, width = image.shape[:2]

            # flag : 얼굴 검출할 것인지에 대한 상태 , 이미지를 읽어들인다.
            # True : 얼굴 검출된 영상을 보여줌, false : 영상만 보여줌.
            if self.flag:
                image = self.face_recog.get_frame(image, self.mosaic_flag)
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # BGR to RGB
            # opencv는 기본적으로 이미지를 불러올 때 BGR(Blue, Green, Red 순으로 받아옴.)
            # RGB로 바꾸어 주어야만 영상이 제대로 나옴.
            self.VideoSignal.emit(QImage(color_swapped_image.data, width, height, color_swapped_image.strides[0], QImage.Format_RGB888))

            loop = QEventLoop()
            QTimer.singleShot(25, loop.quit)  # 25 ms
            loop.exec_()

    # click "BTN2" event, callback => face recognition
    # BTN2를 누르면 flag true or false로 상태 변환 (얼굴을 검출할 것인지 결정)
    @pyqtSlot()
    def face_detection(self):
        self.flag = ~self.flag

    # click "BTN3" event, callback  => mosaic flag set
    # BTN3을 누르면 mosaic_flag true or false로 상태 변환 (모자이크를 할 것인지 결정)
    @pyqtSlot()
    def mosaic(self):
        self.mosaic_flag = ~self.mosaic_flag


# 쓰레드 작업2
class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QImage()
        self.setAttribute(Qt.WA_OpaquePaintEvent) #더블버퍼링 최적화

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QImage()

    def initUI(self):
        self.setWindowTitle("")

    @pyqtSlot(QImage)
    def setImage(self, image):
        if image.isNull():
            pass

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


# widget setting for main window
class MainWidget(QWidget):
    def __init__(self):
        super().__init__(parent=None)

        # label
        manual_label = [QLabel("HUMIC"), QLabel("%s" % name)]


        for i in range(0, 2):
            manual_label[i].setStyleSheet("Color : gray")
            manual_label[i].setFont(QFont("", 60 - (24 * i)))

        # horizontal layout
        hbox = [QHBoxLayout(), QHBoxLayout()]
        for i in range(0, 2):
            hbox[i].addStretch(1)
            hbox[i].addWidget(manual_label[i])
            hbox[i].addStretch(1)

        self.btn_db = QPushButton("%s이미지 등록하기%s" % (" " * 6, " " * 6))
        self.btn_db.setFont(QFont("", 30))
        self.btn_start = QPushButton("%sStart%s" % (" " * 12, " " * 12))
        self.btn_start.setFont(QFont("", 30))

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox[0])
        vbox.addLayout(hbox[1])
        vbox.addStretch(1)

        hbox1, hbox2 = QHBoxLayout(), QHBoxLayout()
        hbox1.addStretch(1)
        hbox1.addWidget(self.btn_db)
        hbox1.addStretch(1)

        hbox2.addStretch(1)
        hbox2.addWidget(self.btn_start)
        hbox2.addStretch(1)

        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addStretch(1)
        self.setLayout(vbox)


class MainWindow(QMainWindow):
    def __init__(self, cur, con):
        self.thread = QThread()
        self.vid = None
        self.image_viewer = None
        self.main_window2 = None

        super().__init__(parent=None)
        self.cursor = cur  # sqlite cursor
        self.con = con
        # if not exist face table, create table
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS FACE (name TEXT, data BLOB)""")

        wg = MainWidget()
        self.setCentralWidget(wg)
        self.resize(800, 600)
        self.setWindowTitle("HUMIC")
        self.show()
        wg.btn_db.clicked.connect(self.func_db)
        wg.btn_start.clicked.connect(self.func_camera)

    def func_db(self):
        fname = QFileDialog.getOpenFileName(self, "choose only image files")  # file dialog
        try:
            name_list = fname[0].split("/")
            face_name = name_list[-1]
            with open(fname[0], "rb") as f:
                bin = f.read()  # binary
            # insert image
            self.cursor.execute("""INSERT INTO FACE (name, data) VALUES (?, ?)""", (face_name, bin))
            self.con.commit()
        except:
            print("no file")

    def func_camera(self):
        self.vid = ShowVideo(cur, con)
        self.vid.face_recog.call_face()
        self.main_window2 = MainWindow2(self)  # second page instance
        self.main_window2.working()
        self.main_window2.btn_back.clicked.connect(self.back_main)
        self.main_window2.btn_face.clicked.connect(self.vid.face_detection)
        self.main_window2.btn_mos.clicked.connect(self.vid.mosaic)
        self.vid.VideoSignal.connect(self.image_viewer.setImage)
        self.vid.startVideo()

    # come back
    def back_main(self):
        self.vid.run_video = False
        self.show()
        self.main_window2.close()


class MainWindow2(QMainWindow):
    def __init__(self, previous_instance):
        super(MainWindow2, self).__init__()
        # GUI generation and option
        self.cursor = previous_instance
        self.btn_face = QPushButton("범인검거용")
        self.btn_face.setFont(QFont("", 15))
        self.btn_mos = QPushButton("방송용")
        self.btn_mos.setFont(QFont("", 15))
        self.btn_back = QPushButton("Back")
        self.btn_back.setFont(QFont("", 15))

        # widget
        vertical_layout = QVBoxLayout()
        horizontal_layout = QHBoxLayout()
        previous_instance.image_viewer = ImageViewer()
        horizontal_layout.addWidget(previous_instance.image_viewer)
        hbox = QHBoxLayout()
        vertical_layout.addLayout(horizontal_layout)  # 레이아웃
        hbox.addWidget(self.btn_face)
        hbox.addWidget(self.btn_mos)
        hbox.addWidget(self.btn_back)

        vertical_layout.addLayout(hbox)

        layout_widget = QWidget()
        layout_widget.setLayout(vertical_layout)

        # 2nd window
        self.setCentralWidget(layout_widget)
        self.setFixedSize(800, 600)
        self.setWindowTitle("HUMIC")
        self.show()
        previous_instance.close()

        exit = QAction("Quit", self)
        exit.triggered.connect(self.closeEvent)

    def closeEvent(self, _):
        if self.cursor.vid.run_video:
            sys.exit()

    def working(self):
        self.cursor.thread.start()
        self.cursor.vid.run_video = True
        self.cursor.vid.moveToThread(self.cursor.thread)


if __name__ == "__main__":
    warnings.filterwarnings(action="ignore")

    # sqlite
    con = sqlite3.connect("./image.db")  # 데이터 베이스 연결
    cur = con.cursor()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(cur, con)
    window.setFixedSize(800, 600)
    sys.exit(app.exec_())