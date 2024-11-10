import cv2
from ultralytics import YOLO
import easyocr
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse

parser = argparse.ArgumentParser(description = 'License plate reader')
parser.add_argument('--path',dest='path',type = str, default = './license_plate_detector.pt',
                    help = 'yolov8 weights path')
parser.add_argument('--source',dest='source',type = str, default = './sample.mp4',
                    help = 'video file path')
args = parser.parse_args()

weight_path = args.path
source = args.source

reader = easyocr.Reader(['en'], gpu=False)
plate_model = YOLO(weight_path)
results = {}
conf_threshold = 0.5
tracker = DeepSort(max_age = 30)

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')
        return text, score

    return None, None

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{}\n'.format('frame_nmr','license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                        'license_number_score'))

        for frame_nmr in results.keys():
            print(results[frame_nmr])
            if 'license_plate' in results[frame_nmr].keys() and \
                'text' in results[frame_nmr]['license_plate'].keys():
                f.write('{},{},{},{},{}\n'.format(frame_nmr,
                                                        '[{} {} {} {}]'.format(
                                                            results[frame_nmr]['license_plate']['bbox'][0],
                                                            results[frame_nmr]['license_plate']['bbox'][1],
                                                            results[frame_nmr]['license_plate']['bbox'][2],
                                                            results[frame_nmr]['license_plate']['bbox'][3]),
                                                        results[frame_nmr]['license_plate']['bbox_score'],
                                                        results[frame_nmr]['license_plate']['text'],
                                                        results[frame_nmr]['license_plate']['text_score'])
                        )
        f.close()

if __name__ == '__main__':
    cap = cv2.VideoCapture(source)
    frame_nmr = -1
    ret = True
    tracked = []
    i = 1
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            license_plates = plate_model(frame,verbose=False)[0]
            detect = []
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                if score < conf_threshold:
                    continue
                detect.append([[x1,y1,x2-x1,y2-y1],score,class_id])

            tracks = tracker.update_tracks(detect, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1,y1,x2,y2 = map(int,ltrb)
                
                label = '{}-{}'.format('plate',track_id)

                frame = cv2.rectangle(frame, (x1,y1), (x2,y2),(255,255,0), thickness=2)
                # frame = cv2.rectangle(frame, (x1-1,y1-20), (x1+len(label)*12,y1),(255,255,0), thickness=-1)
                frame = cv2.putText(frame, label, (x1+5,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0), thickness=2)

                if track_id not in tracked:
                    tracked.append(track_id)
                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    print('license plate',i,':',license_plate_text)
                    i += 1
                    # if license_plate_text is not None:
                    #     results[frame_nmr] = {'license_plate': {'bbox': [x1, y1, x2, y2],
                    #                                                     'text': license_plate_text,
                    #                                                     'bbox_score': score,
                    #                                                     'text_score': license_plate_text_score}}
            frame = cv2.resize(frame, (1080, 629))
            cv2.imshow('A',frame)
            key = cv2.waitKey(1) 
            if key == ord("q"): 
                break

    cap.release()
    cv2.destroyAllWindows()
    # write_csv(results, './test.csv')