import ultralytics as ultralytics
import cv2
import os

model = ultralytics.YOLO('yolov8n.pt')

test_dataset_path = 'Dataset/test'

correct = 0
total = 0
for folder_name in os.listdir(test_dataset_path):
    class_label = folder_name

    for image_path in os.listdir(os.path.join(test_dataset_path, folder_name)):
        image = cv2.imread(os.path.join(test_dataset_path, folder_name, image_path))

        results = model.predict(image)
        label = results[0].boxes.cls
        names = results[0].names
        total += 1
        try:
            if len(names) == 1:
                if class_label.lower() == str(names[int(label)]).lower():
                    correct += 1
                print(f"Predicted: {names[0][int(label)]}, Actual: {class_label}")
            elif len(names) == 2:
                if class_label.lower() == str(names[0][int(label)]).lower():
                    correct += 1
                print(f"Predicted: {names[0][int(label)]}, Actual: {class_label}")
            else:
                pass
        except:
            pass
                
print(f"Accuracy: {correct/total}")