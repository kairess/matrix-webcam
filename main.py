import cv2

cap = cv2.VideoCapture(0)
bg_cap = cv2.VideoCapture('bg/bg3.mp4')
'''
Video source:
https://www.youtube.com/watch?v=WyKflRx56Tg
https://www.youtube.com/watch?v=cHhGjvf4jBI
https://www.youtube.com/watch?v=Dn56rvuPGZU
https://www.youtube.com/watch?v=hwSp3VYR_2o
https://www.motionstock.net/free-videos/matrix-digital-rain/
'''

cap_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), cap_size)

sub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=100, detectShadows=False)

while cap.isOpened():
    ret, fg_img = cap.read()

    if not ret:
        break

    bg_ret, bg_img = bg_cap.read()

    if not bg_ret:
        bg_cap.set(1, 0)
        _, bg_img = bg_cap.read()

    bg_img = cv2.resize(bg_img, dsize=cap_size)

    mask = sub.apply(fg_img)

    # https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    result = cv2.bitwise_and(bg_img, fg_img, mask=mask)

    # cv2.imshow('fg', fg_img)
    # cv2.imshow('bg', bg_img)
    # cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    out.write(result)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
bg_cap.release()
