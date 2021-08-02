import cv2
import dlib
from imutils import face_utils
import time

def face_detector():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(0)

    while True:
        ret,frame = cap.read()
        start_time = time.time()

        faces = detector(frame, 0)
        for i, d in enumerate(faces):
            shape = predictor(frame, d)
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        end_time = time.time()
        time_diff = end_time - start_time
        fps = 1 / time_diff
        fps_text = 'fps : {:2f}'.format(fps)
        cv2.putText(frame, fps_text, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('frame',frame)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    face_detector()