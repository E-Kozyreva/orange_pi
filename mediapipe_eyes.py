import ctypes
from queue import Empty
from time import sleep
# import datetime
# import numpy as np
# import itertools
from multiprocessing import Queue, Value
from multiprocessing import Process
import time
import mediapipe as mp
import cv2



IMG_SIZE = 640
RKNN_MODEL = ''


def FaceLandmarker(queue, res_queue, KeepAlive):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
        while KeepAlive.value == True:
            try:
                item = queue.get()
                image = cv2.cvtColor(item[0], cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                res_queue.put([bool(results.multi_face_landmarks), item[1]])
            except Empty:
                print(f'Queue mp : {queue.qsize()}', flush=True)
                sleep(0.3)
                continue
            except KeyboardInterrupt:
                break

def ImageGraber(queue, KeepAlive):
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)

    i = int(0)
    
    while (cap.isOpened() and KeepAlive.value):
        try:
            success, img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            queue.put([img, i])
            print(f'Queue cv : {queue.qsize()}', flush=True)
            if(i == 59):
                i = int(0)
            else:
                i=int(i + 1)
            
        except KeyboardInterrupt:
            break
    cap.release()

    
def CollectAIResult(queue, KeepAlive):
    t1 = time.time()
    while KeepAlive.value == True:
        try:
            i = queue.get(timeout=0.3)
            print(f'{i[1]} {i[0]}')
            if i[1] == 59:
                t2 = time.time()
                d = t2-t1
                print(f'FPS: {1/(d/60)}')
                t1 = time.time()
        except Empty :
            print("wait result")
            continue
        except KeyboardInterrupt:
            break

def DetectSign(queue, res_queue, KeepAlive):
    
    rknn_lite = RKNNLite() #init rknn object

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    # print('done')


    # init runtime environment
    # print('--> Init runtime environment')
    ret = rknn_lite.init_runtime()
    # ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    # print('done')


    while KeepAlive.value == True:
        try:
            item = queue.get(timeout=0.1)
            img = item[0]

            # Inference
            # print('--> Running model')
            outputs = rknn_lite.inference(inputs=[img])
            res_queue.put((outputs, item[1]))
        except Empty:
            # sleep(0.1)
            continue        
        except KeyboardInterrupt:
            break

    rknn_lite.release()


if __name__ == '__main__':
    # mp_face_mesh = mp.solutions.face_mesh
    Count_AI_proc = int(3)
    try:
        KeepAlive = Value(ctypes.c_bool, True)  #shared value for stop processes
        queue = Queue()         #create image queue (from ImageGraber to FaceLandmarker)
        res_queue = Queue()     #create eye_status queue (from FaceLandmarker to CollectAIResult)

        process_list = []
        process_list.append(Process(target=ImageGraber, args=(queue, KeepAlive)))
        process_list.append(Process(target=CollectAIResult, args=(res_queue, KeepAlive)))
        for _ in range(Count_AI_proc):
            process_list.append(Process(target=FaceLandmarker, args=(queue, res_queue, KeepAlive)))
            #process_list.append(Process(target=DetectSign, args=(queue, res_queue, KeepAlive)))
            
        
        for proc in process_list:
            proc.start()

        for proc in process_list:
            proc.join()


    except KeyboardInterrupt:
        print("Stop")
        #KeepAlive.value = False
        #for proc in process_list:
        #    proc.join()
    
