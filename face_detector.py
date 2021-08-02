import cv2
import dlib
import datetime
import time

def face_detector():
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)


    detector = dlib.get_frontal_face_detector()

    while True:
        ret,frame = cap.read()
        start_time = time.time()
        face_recs,scores,idxs = detector.run(frame,0)

        for i,d in enumerate(face_recs):
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            text = 'score : {:4.2f}'.format(scores[i])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,text,(x1,y1-10),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,0,255),2)

        end_time = time.time()
        time_diff = end_time - start_time

        fps = 1/time_diff
        fps_text = 'fps : {:2f}'.format(fps)
        cv2.putText(frame,fps_text,(20,20),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,255,0),2)

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    face_detector()