from threading import Thread, Lock, get_ident
import cv2

class CamVideoStream :

    def __init__(self, src = 0, width = 320, height = 240 ,path="./",filename="CCTV.mp4",fps=15, codec=cv2.VideoWriter_fourcc(*'mp4v')) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.recoder_started = False
        self.path = path+""+filename
        self.fps = fps
        self.codec = codec
        self.read_lock = Lock()
    

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def start_recoder(self):
        self.stream_recoder = cv2.VideoWriter(str(self.path), self.codec, self.fps, (int(self.width), int(self.height)))
        self.recoder_started = True
        self.recoder_thread = Thread(target=self.write, args=())
        self.recoder_thread.start()

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            # self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            # self.read_lock.release()
            
    def write(self) :
        while self.recoder_started:
            (grabbed, frame) = self.stream.read()
            # self.read_lock.acquire()
            self.stream_recoder.write(frame)
            # self.read_lock.release()
                
    def read(self) :
        # self.read_lock.acquire()
        frame = self.frame.copy()
        # self.read_lock.release()
        return frame

    def stop_recorder(self) :
        self.recoder_started = False
        if self.recoder_thread.is_alive():
            self.recoder_thread.join()
        self.stream_recoder.release()

    def stop(self) :
        self.started = False
        self.recoder_started = False
        if self.thread.is_alive():
            self.thread.join()
        if self.recoder_thread.is_alive():
            self.recoder_thread.join()
            

    def show(self):
        while True :
            cv2.imshow('frame '+str(get_ident()), self.frame)
            if cv2.waitKey(1) == ord('q'):   
                self.stop()
                break
        self.exit()

    def is_stopped(self):
        return self.started 

    def exit(self) :
        self.stream.release()
        self.stream_recoder.release()
