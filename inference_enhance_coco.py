# Using a pretrained model to enhance existing COCO dataset. 

import pickle
import json
import time
import typing as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from mmdet.apis import DetInferencer
import mmcv
import numpy as np
from tqdm import tqdm


COCO_CATEGORIES_TO_NEW_ONES = {
    2:7,  # car
    3:8,  # motorcycle
    5:9,  # bus
    7:10, # truck
    0:11  # person
}
NEW_CATEGORIES_NAMES = {
    7:  "car",
    8:  "motorcycle",
    9:  "bus",
    10: "truck",
    11: "person"
}
SUPERCATEGORY = "Enhancement"
    

# TODO: move these somewhere proper
config_file = 'configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'
checkpoint_file = 'https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_r50_8xb2-lsj-50e_coco/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth'




def parse_result_to_coco(predictions:dict, image_id:int, annotation_id_start:int, confidence_threshold=0.6, 
                         epsilon=0.001)->T.List[dict]:
    ''' parse MMDet DetInferencer prediction for a single image into COCO-formatted dictionary 
        args:
            - prediction: dictionary containing 'labels', 'scores', 'bboxes', 'masks' lists of the corresponding length
            - image_id: id of the current image
            - annotation_id_start: has to now where to start indexing annotations --global across all images
            - epsilon: (relative) OpenCV approxPolyDP: Parameter specifying the approximation accuracy. This is the maximum distance between the original curve and its approximation.
        returns:
            - list of coco annotations
    '''
    
    labels = predictions['labels']
    scores = predictions['scores']
    bboxes = predictions['bboxes']
    masks  = predictions['masks']
    coco_annotations = [] # all coco annotations for the image
    
    current_annot_id = annotation_id_start
    for idx, label in enumerate(labels):
        if label not in COCO_CATEGORIES_TO_NEW_ONES or scores[idx] < confidence_threshold:
            continue
        
        # fit polygon annotation to mask COCO style
        mask = masks[idx].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
       
        # if len(contours) > 1:
        #     mask = predictions['masks'][idx].astype(np.uint8)
        #     plt.imsave('mask_with_multiple_contours.jpg', mask)
        #     print([f"l={cv2.arcLength(cont, True)} area={cv2.contourArea(cont)}" for cont in contours])
        #     print("more contours!")

        # single mask might contain multiple disconected regions, and I believe multi-region COCO segmentation are not supported (right now) --although the segmentation field is a 2D list
        # split multi-region segmentations into a multiple separate annotations?
        # for now, just keep the one with largest area
        contours = [contours[np.argmax([cv2.contourArea(cont) for cont in contours])]]
        assert len(contours) == 1
        
        coco_polygons = [] 
        for cont in contours:
            eps_abs = epsilon * cv2.arcLength(cont, True) # get absolute epsilon by scaling with the perimeter of the closed contour
            cont_simpl = cv2.approxPolyDP(cont, eps_abs, True)
            coco_polygons.append(cont_simpl.flatten().tolist())
            print(f"countour length: original: {len(cont)}, simplified: {len(cont_simpl)}")

        bbox = bboxes[idx]
        coco_bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

        coco_annotations.append({
            "id": current_annot_id,
            "image_id": image_id,
            "category_id": COCO_CATEGORIES_TO_NEW_ONES[label],
            "bbox": coco_bbox,
            "area": coco_bbox[2] * coco_bbox[3],
            "segmentation": coco_polygons,
            "iscrowd": 0
        })
        current_annot_id += 1

    return coco_annotations 

def detect_and_enhance_coco(img_dir, 
                            annotations_file='_annotations.coco.json', 
                            annotations_output_file='_annotations_enhanced.coco.json'):
    ''' Enhance coco annotations by detections given MMDET model'''
    
    total_inference_time = 0.0
    with open(Path(img_dir)/annotations_file) as f:
        annotations = json.load(f)

    # update categories
    for id, name in NEW_CATEGORIES_NAMES.items():
        annotations['categories'].append({
            "id": id,
            "name": name,
            "supercategory": SUPERCATEGORY
        })

    annotation_id = annotations['annotations'][-1]['id'] + 1 # init global id

    inferencer = DetInferencer(model=config_file, weights=checkpoint_file, device="cuda:0", show_progress=False)
    for image_annot in tqdm(annotations['images']):
        img_p = Path(img_dir)/image_annot['file_name']
        assert img_p.exists()
        
        img = mmcv.imread(str(img_p))
        t = time.monotonic()
        prediction_result = inferencer(img)
        total_inference_time += time.monotonic() - t

        predictions = prediction_result['predictions'][0] # not sure why this produces list, maybe batching? TODO: iterate?
        coco_annotation_enhancement = parse_result_to_coco(predictions, image_annot['id'], annotation_id)
        annotation_id += len(coco_annotation_enhancement)

        annotations['annotations'].extend(coco_annotation_enhancement)
    
    print(f"detection ran on {len(annotations['images'])}, total inference time: {total_inference_time}")
    with open(Path(img_dir)/annotations_output_file, 'w') as f:
        json.dump(annotations, f, indent=4)
    print(f"enhanced annotations save to {Path(img_dir)/annotations_output_file}")



if __name__=='__main__':
    dataset_dir = "/home/mosaicpc-ubuntu/Downloads/mosaic_nick_dataset_rot_with_saudis/valid"

    detect_and_enhance_coco(dataset_dir)