import cv2
from keras.models import load_model

model = load_model('cityscapemodel.h5')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    # print(str(cap.isOpened()))
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    cv2.imshow("Video", test_img)
    resized_img = cv2.resize(test_img, (256, 256))
    reshaped_img = resized_img.reshape(1, 256, 256, 3)
    segmented_img = model.predict(reshaped_img)

    cv2.imshow('Segmented Images', segmented_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
