import cv2

detector_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    ok, frame = video_capture.read()

    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    deteccoes = detector_face.detectMultiScale(imagem_cinza, minSize=(100, 100))

    for x, y, w, h in deteccoes:
        print(w, h)
        cv2.rectangle(frame, (x,y), (x + w , y+ h),(0, 255, 0), 2)

    #Mostrar o resultado    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Libera a memoria
video_capture.release()
cv2.destroyAllWindows()

    
