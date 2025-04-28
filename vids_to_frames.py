import os
import cv2
from tqdm.notebook import tqdm

def vid_to_frames(vid_path, datasets_path):
    vid_filename = os.path.splitext(os.path.basename(vid_path))[0]
    print(vid_filename)

    place = vid_filename[:vid_filename.find("_vid_")]
    vid_dir = vid_filename[vid_filename.find("_vid_") + 1:]
    print(place)
    print(vid_dir)
    
    to_dir = os.path.join(datasets_path, place, vid_dir)
    os.makedirs(to_dir)

    cap = cv2.VideoCapture(vid_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames)
    print(f"Total frames: {total_frames}")            
    frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            filename = f"{place}_{vid_dir}_frame_{frame_num}.png"
            cv2.imwrite(os.path.join(to_dir, filename), frame)
            frame_num += 1
            pbar.update(1)
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    pbar.close()

    return frame_num

vid_dic = {}

vid_path = "vid_path"
datasets_path = "datasets_path"

filename = os.path.basename(vid_path)
frame_num = vid_to_frames(vid_path, datasets_path)
vid_dic[filename] = frame_num

print(vid_dic)



