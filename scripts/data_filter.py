import pickle
import time
# from tqdm import tqdm

# Two person interaction classes
class_list = [
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
    60, 106, 107, 108, 109, 110, 111, 112, 113, 114, 
    115, 116, 117, 118, 119, 120]

# remap = {50:0,51:1:}

ntu_60 = '/storage/datasets/kitti/poseview/original/ntu60_hrnet.pkl'
ntu_120 = '/storage/datasets/kitti/poseview/original/ntu120_hrnet.pkl'

with open(ntu_120, 'rb') as f:
    Load1 = pickle.load(f)
    
annotations = Load1['annotations']
split = Load1['split']

two_person_annotations = []

print(Load1.keys())
print(len(annotations))
# 56578

# Based on subject
# xsub_train 40091
# xsub_val 16487

# Based on camera view
# xview_val 18932 
# xview_train 37646


# Labels are 0 based
# frame dirs are 1 based 
for i in range(len(annotations)):
    # if int(annotations[i]['label']) == 1:
    #     print(annotations[i]['frame_dir'])
    if int(annotations[i]['label']) in class_list:
        if annotations[i]['label']<=60:
            annotations[i]['label'] = annotations[i]['label'] - 50
        else:
            annotations[i]['label'] = annotations[i]['label'] - 95

        two_person_annotations.append(annotations[i])
        # print(annotations[i].keys())
        
print(len(two_person_annotations))
# 9406


print("Frane dir: ",two_person_annotations[0]['frame_dir'])
print("label: ",two_person_annotations[0]['label'])
print("image_shape: ",two_person_annotations[0]['img_shape'])
print("original image shape: ",two_person_annotations[0]['original_shape'])
print("total frames: ",two_person_annotations[0]['total_frames'])
print("keypoints: ",two_person_annotations[0]['keypoint'].shape)
print("keypoint_score: ",two_person_annotations[0]['keypoint_score'].shape)

print(two_person_annotations[0].keys())

xsub_train = len(split['xsub_train'])
xsub_val = len(split['xsub_val'])

# print(Load1['split']['xsub_val'])
print(Load1['split'].keys())

print(xsub_train, xsub_val)

print("train split :",xsub_train/(xsub_train + xsub_val))
print("test split :",xsub_val/(xsub_train + xsub_val))


two_person_split = dict(xsub_train = [], xsub_val = [], xview_train = [], xview_val = [])

two_person_split['xsub_train'] = [item for item in split['xsub_train'] if int(item[-3:]) in class_list]
two_person_split['xsub_val'] = [item for item in split['xsub_val'] if int(item[-3:]) in class_list]
two_person_split['xset_val'] = [item for item in split['xset_val'] if int(item[-3:]) in class_list]
two_person_split['xset_train'] = [item for item in split['xset_train'] if int(item[-3:]) in class_list]

xsub_train = len(two_person_split['xsub_train'])
xsub_val = len(two_person_split['xsub_val'])

print("train split :",xsub_train/(xsub_train + xsub_val))
print("test split :",xsub_val/(xsub_train + xsub_val))

train_classes = set()
test_classes = set()

for item in two_person_split['xsub_train']:
    train_classes.add(item[-3:])

for item in two_person_split['xsub_val']:
    test_classes.add(item[-3:])
    
print("train: ",train_classes)
print("test: ",test_classes)

new_file = dict(split=two_person_split, annotations=two_person_annotations)

print(new_file.keys())

with open('ntu120_two.pkl', 'wb') as file:
    pickle.dump(new_file, file)