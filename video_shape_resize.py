import cv2

INPUT = 'test.mp4'
OUPUT = 'out.avi'
OUPUT_SHAPE = (640, 640)
FPS = 30


vid_capture = cv2.VideoCapture(INPUT)
output = cv2.VideoWriter(OUPUT,
                          cv2.VideoWriter_fourcc('M','J','P','G'), FPS, OUPUT_SHAPE)


while(vid_capture.isOpened()):
    # Метод vid_capture.read() возвращает кортеж, первым элементом которого является логическое значение
    # а вторым - кадр
    ret, frame = vid_capture.read()
    if ret == True:
       # Записываем фрейм в выходные файлы
       frame = cv2.resize(frame,OUPUT_SHAPE)
       output.write(frame)
       print('Convert!')
    else:
       print('Поток отключен')
       break
           
vid_capture.release()
output.release()