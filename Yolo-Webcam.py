from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)


model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

cell_object_width= 15
cell_d = 46
bottle_object_width = 35
bottle_d = 46
kb_object_width = 30.75
kb_d = 46
book_object_width = 14
book_d = 46


prev_frame_time = 0
new_frame_time = 0

# while True:
#     new_frame_time = time.time()
#     success, img = cap.read()
#     results = model(img, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             print("Bounding Box Coordinates: ", (x1, y1, x2, y2))
#             # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#             w, h = x2 - x1, y2 - y1
#             cvzone.cornerRect(img, (x1, y1, w, h))
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])
#
#
#             cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
#
#     fps = 1 / (new_frame_time - prev_frame_time)
#     prev_frame_time = new_frame_time
#     print(fps)
#
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1,y1,x2,y2)
            w, h = x2 - x1, y2 - y1
            class_index = int(box.cls[0])

            # Check if the detected class is a cell phone
            if classNames[class_index] == "cell phone":
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cell_focal_length = (cell_object_width * cell_d) / w
                cell_apparent_width = w  # Apparent width of the cell phone in pixels
                cell_distance = ((cell_object_width * 3.9) / cell_apparent_width)*100
                cvzone.putTextRect(img, f'cell phone, distance: {int(cell_distance)}cm', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            elif classNames[class_index] == "bottle":
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                bottle_focal_length = (bottle_object_width * bottle_d) / w
                #print("bottle focal: ",bottle_focal_length)
                bottle_apparent_width = w  # Apparent width of the cell phone in pixels
                bottle_distance = ((bottle_object_width * 16.75) / bottle_apparent_width)*9.2
                cvzone.putTextRect(img, f'bottle, distance: {int(bottle_distance)}cm', (max(0, x1), max(35, y1)), scale=1,thickness=1)
            elif classNames[class_index] == "keyboard":
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                kb_focal_length = (kb_object_width * kb_d) / w
                # print("keyboard focal: ",kb_focal_length)
                kb_apparent_width = w  # Apparent width of the cell phone in pixels
                kb_distance = ((kb_object_width * 2.65) / kb_apparent_width)*291
                cvzone.putTextRect(img, f'keyboard, distance: {int(kb_distance)}cm', (max(0, x1), max(35, y1)),scale=1, thickness=1)
            elif classNames[class_index] == "book":
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                book_focal_length = (book_object_width * book_d) / w
                #print("book focal: ",book_focal_length)
                book_apparent_width = w  # Apparent width of the cell phone in pixels
                book_distance = ((book_object_width * 1.69) / book_apparent_width)*901
                cvzone.putTextRect(img, f'book, distance: {int(book_distance)}cm', (max(0, x1), max(35, y1)),scale=1, thickness=1)


            else:
                cls = int(box.cls[0])
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(35, y1)), scale=1, thickness=1)




    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

