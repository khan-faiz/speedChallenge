import time
import cv2
import os
import json
import csv

# change these depending on your file names / paths
#TEST_GROUND_TRUTH_JSON_PATH = './data/drive.json' # change this to the test ground truth
#VIDEO_PATH = './data/drive.mp4' # change this to the test video
#TEST_GROUND_TRUTH_JSON_PATH = './../data/train.json' # change this to the test ground truth
#VIDEO_PATH = './../data/train.mp4' # change this to the test video
TEST_GROUND_TRUTH_JSON_PATH = './../data/test.json' # change this to the test ground truth
VIDEO_PATH = './../data/test.mp4' # change this to the test video

with open(TEST_GROUND_TRUTH_JSON_PATH) as json_data:
    ground_truth = json.load(json_data)
    # json_data.close()
with open(DRIVE_TEST_CSV_PATH, 'w') as csvfile:
    fieldnames = ['image_path', 'time', 'speed']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FRAME_COUNT, len(ground_truth))
#     cap.set(cv2.CAP_PROP_FPS, 11.7552) #11.7552


    for idx, item in enumerate(ground_truth):
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        # read in the image
        success, image = cap.read()
        if success:
            image_path = os.path.join(TEST_IMG_PATH, str(item[0]) + '.jpg')
            
            # save image to IMG folder
            cv2.imwrite(image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY),100])
            time.sleep(0.75)
            print('wrote img', idx)
            
            # write row to driving.csv
            writer.writerow({'image_path': image_path, 
                     'time':item[0],
                     'speed':item[1],
                    })

print('done writing to driving_test.csv and test_IMG folder')
