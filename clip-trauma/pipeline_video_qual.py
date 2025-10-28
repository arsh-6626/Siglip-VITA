import os
import warnings
from argparse import ArgumentParser
import torchvision.transforms as transforms
import cv2
import numpy as np
from infer_fast_testing import get_limbs
from models.vqvae import VQVAE
from PIL import Image
import torch
from ultralytics import YOLO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
from collections import deque
from new_model import NewModel
from infer_trauma_cls import classify,classify_list
from dt_apriltags import Detector
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_keypoints_and_skeleton(image, keypoints):
    
    BODY_PART_COLORS = {
    'nose': (0, 255, 0),          # Green
    'left_eye': (0, 0, 255),      # Red
    'right_eye': (255, 0, 0),     # Blue
    'left_ear': (0, 255, 255),    # Yellow
    'right_ear': (255, 255, 0),   # Cyan
    'left_shoulder': (255, 0, 255),  # Magenta
    'right_shoulder': (128, 128, 128), # Gray
    'left_elbow': (255, 165, 0),  # Orange
    'right_elbow': (0, 128, 0),   # Dark Green
    'left_wrist': (128, 0, 128),  # Purple
    'right_wrist': (0, 128, 128), # Teal
    'left_hip': (255, 192, 203),  # Pink
    'right_hip': (128, 128, 0),   # Olive
    'left_knee': (0, 255, 128),   # Light Green
    'right_knee': (255, 105, 180), # Hot Pink
    'left_ankle': (255, 140, 0),  # Dark Orange
    'right_ankle': (255, 69, 0)    # Red-Orange
    }

    # Body part names corresponding to the array indices
    BODY_PART_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # Define skeleton connections (pairs of indices)
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5,7),(7,9),(6,8),(8,10),(5, 11),
        (6, 12), (11, 12), (11, 13), (12, 14), (13, 15),
        (14, 16), (15, 17) ]
        
    """
    Draw keypoints and skeleton on the image.

    :param image: Input image
    :param keypoints: Array of keypoints with (x, y) coordinates
    """
    # Draw keypoints
    
    for i, color in enumerate(BODY_PART_COLORS.values()):
        if i < len(keypoints):
            x, y = keypoints[i]
            image_new = cv2.circle(image, (int(x), int(y)), 5, color, -1)  # Draw keypoints
    
    # Draw skeleton connections
    for (i, j) in SKELETON_CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            x1, y1 = keypoints[i]
            x2, y2 = keypoints[j]
            color = BODY_PART_COLORS[BODY_PART_NAMES[i]]  # Use color of the first part
            image_new = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Draw skeleton lines

    return image_new

def show_keypoints(image,keypoints):
    # Convert keypoints to integer coordinates
    keypoints = keypoints.astype(int)

    # Read the image
    # Define the pairs of points to connect
    pairs = [
                (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 11),
                (6, 12), (11, 12), (11, 13), (12, 14), (13, 15),
                (14, 16), (15, 16)
            ]

    # Draw lines between the specified keypoints
    for pair in pairs:
        pt1 = tuple(keypoints[pair[0]])
        pt2 = tuple(keypoints[pair[1]])
        image = cv2.line(image, pt1, pt2, color=(0, 255, 0), thickness=2)
    return image

def resize_or_pad_image(img, target_size):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # Calculate aspect ratio
    aspect_ratio = w / h
    
    # If either dimension is greater than the target size, resize while maintaining aspect ratio
    if w > target_w or h > target_h:
        if aspect_ratio > 1:  # Width is greater than height
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else:  # Height is greater or equal to width
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_h = h
        new_w = w
    # Calculate padding
    h, w = img.shape[:2]
    top = (target_h - h) // 2
    bottom = target_h - h - top
    left = (target_w - w) // 2
    right = target_w - w - left
    
    # Add padding
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return img , (top,bottom,left,right), (new_h,new_w,aspect_ratio)

def rgb_to_bgr(image):
    # Convert a PIL image from RGB to BGR
    image_array = np.array(image)
    bgr_image_array = image_array[..., ::-1]
    return Image.fromarray(bgr_image_array)

def tensor_to_pil(tensor):
    image = tensor.squeeze(0).cpu().detach().clamp(0, 1).permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

def reconstruct_and_visualize(image, model, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        image_tensor = image.to(device)
        vq_encoder_output = model.pre_quantization_conv(model.encoder(image_tensor))
        _, z_q, _, _, e_indices = model.vector_quantization(vq_encoder_output)
        x_recon = model.decoder(z_q)
        original_image = tensor_to_pil(image_tensor)
        reconstructed_image = tensor_to_pil(x_recon)
        reconstructed_image = reconstructed_image.resize(original_image.size)
        # print(reconstructed_image)
        original_images = tensor_to_pil(image_tensor)
        # original_images.show(title="Original Images")
        original_width, original_height = original_images.size
        reconstructed_width, reconstructed_height = reconstructed_image.size

        total_width = original_width + reconstructed_width
        max_height = max(original_height, reconstructed_height)
        combined_image = Image.new('RGB', (total_width, max_height))

        # Paste the images into the combined image
        combined_image.paste(original_images, (0, 0))
        combined_image.paste(reconstructed_image, (original_width, 0))
        combined_image=rgb_to_bgr(combined_image)

        # Show the combined image
        combined_image.show(title="Combined Image")
                # reconstructed_image.show(title="Reconstructed Image")
        # print(z_q.shape)

def load_model(model_filename):
    path = ""
    
    if torch.cuda.is_available():
        data = torch.load(path + model_filename)
    else:
        data = torch.load(path+model_filename,map_location=lambda storage, loc: storage)
    
    # params = data["hyperparameters"]
    
    model = VQVAE(128, 32,
                  2, 512, 
                  64, 0.23).to(device)

    model.load_state_dict(data['model'])
    
    return model, data

def bfs(image, startrow,startcol, visited):
    # Create a queue for BFS
    q = deque()
    row,col = image.shape
    max_x, min_x, max_y, min_y = 0, row, 0 ,col 
    # Mark the current node as visited and enqueue it
    visited[startrow][startcol] = True
    q.append([startrow,startcol])

    # Iterate over the queue
    while q:
        # Dequeue a vertex from queue and print it
        currentnode = q.popleft()
        currentrow,currentcol = currentnode
        color = image[currentrow, currentcol]
        if color!=0:
            if currentrow>max_x:
                max_x = currentrow
            if currentrow<min_x:
                min_x = currentrow
            if currentcol>max_y:
                max_y = currentcol
            if currentcol<min_y:
                min_y = currentcol

        # Get all adjacent vertices of the dequeued vertex
        # If an adjacent has not been visited, then mark it visited and enqueue it
        for i in range(-1,2):
            for j in range(-1,2):
                if currentrow-i>=0 and currentrow-i<row and currentcol-j>=0 and currentcol-j<col:
                    if not visited[currentrow-i][currentcol-j]:
                        visited[currentrow-i][currentcol-j] = True
                        q.append([currentrow-i,currentcol-j])
    return max_x, max_y, min_x, min_y

def postprocess_limb(limb_list):
    img_cropped_list = []
    for img in limb_list:
        img_resized = cv2.cvtColor(cv2.resize(img, (img.shape[1]//8, img.shape[0]//8)), cv2.COLOR_BGR2GRAY)
        # print(img_resized.shape)
        visited= np.zeros(img_resized.shape, dtype=bool)
        max_x, max_y, min_x, min_y = bfs(img_resized ,0,0, visited)
        image_cropped = cv2.resize(img[8*min_x:8*max_x,8*min_y:8*max_y], (256,256))
        img_cropped_list.append(image_cropped)
    return img_cropped_list

def apriltag_detector(image,Detector):
    tag_results = []
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    color_img = cv2.cvtColor(gray_image,cv2.COLOR_GRAY2RGB)
    tags = Detector.detect(gray_image)
    tag_results.append(tags)
    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(image,tuple(tag.corners[idx-1,:].astype(int)),tuple(tag.corners[idx,:].astype(int)),(0,255,0),5)
        # cv2.putText(image,f"ID:{str(tag.tag_id)}",
        #             org=(image.shape[0] -20,image.shape[1]+20),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=1.5,
        #             color=(0,0,255))
    return image,tags
    # print(tags)
    
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

def extract_non_black_region(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a binary mask where non-black pixels are 1 and black pixels are 0
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find the coordinates of non-black pixels
    coords = np.column_stack(np.where(binary_mask > 0))
    
    # Get the bounding box coordinates
    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)
    
    # Extract the region of interest using the bounding box coordinates
    extracted_region = image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    
    return extracted_region

def show_gaussian_on_image(limb_list, ann_image, body_coords, trauma_values, spacing=10):
    # Extract non-black regions from limbs
    extracted_limbs = [extract_non_black_region(limb) for limb in limb_list]
    
    # Find the smallest limb size
    min_height = min(limb.shape[0] for limb in extracted_limbs)
    min_width = min(limb.shape[1] for limb in extracted_limbs)
    
    # Resize all limbs to the smallest size
    resized_limbs = [cv2.resize(limb, (min_width, min_height)) for limb in extracted_limbs]
    
    # Calculate the total height required for all limbs including spacing
    total_limb_height = len(resized_limbs) * min_height + spacing * (len(resized_limbs) - 1)
    available_height = ann_image.shape[0]
    
    # Calculate initial y_offset to center the limbs vertically
    y_offset = max(0, (available_height - total_limb_height) // 2)
    
    for i, limb in enumerate(resized_limbs):
        limb_height, limb_width = limb.shape[:2]
        
        # Calculate x and y coordinates for the limb placement
        x1 = ann_image.shape[1] - limb_width - 30
        y1 = y_offset
        y2 = min(ann_image.shape[0], y1 + limb_height)
        x2 = min(ann_image.shape[1], x1 + limb_width)
        
        # Ensure the coordinates are within the boundaries and valid
        if y2 > y1 and x2 > x1:
            ann_image[y1:y2, x1:x2] = limb[:y2-y1, :x2-x1]
            
            # Add the trauma text beside the image
            text = f"Trauma: {trauma_values[i].capitalize()}"
            no_text = ""
            # if i==2:
            #     text = f"Trauma: Present"
            # text = f"Trauma: Absent"
            text_x = x1 - 300  # Adjust the x position of the text
            text_y = y1 + limb_height // 2  # Center the text vertically with the image
            cv2.putText(ann_image, no_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Update y_offset for the next limb placement with some spacing
        y_offset += limb_height + spacing
    
    return ann_image

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

person_model=YOLO('best_body.pt')
model_classify = NewModel()
model_classify.to('cuda')
state_dict = torch.load("model_epoch_140.pth")
model_classify.load_state_dict(state_dict)
model_classify.eval()
print("Person Detection Model Loaded")


pose_model = init_pose_model(
       '/home/uas-dtu/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py', 'vitpose_small.pth', 'cuda:0')

print("Pose Detection Model Loaded")

input_video_path = '/home/uas-dtu/trauma_clone_for_testing/ViTPose/demo/output_video.mp4'  # Path to input video
output_video_path = "/home/uas-dtu/trauma_clone_for_testing/ViTPose/demo/output_video_inferred.mp4"

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

transform = transforms.ToTensor()
resize_transform = transforms.Resize((256, 192))

image_number = 0
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # ann_img,tags = apriltag_detector(frame,at_detector)

        ann_img = frame
        height, width, _ = frame.shape
        result_person = person_model.predict(frame,verbose=False,conf=0.1)
        max_area = 0
        max_box = None
        for r in result_person:
            box = r.boxes.xyxy.to("cpu").numpy()
            cls = r.boxes.cls.to("cpu").numpy()
            
            if box.shape[0] == 0:
                break

            # Filter out non-person detections
            person_indices = np.where(cls == 0)[0]
            
            # If there are no person detections, continue
            if len(person_indices) == 0:
                continue

            # Iterate through the person detections to find the box with the max area
            for i in person_indices:
                x1, y1, x2, y2 = box[i]
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    max_box = box[i]
        
        if max_box is not None:

            x1_body, y1_body, x2_body, y2_body = map(int, max_box)
            ann_img = cv2.rectangle(ann_img,(x1_body,y1_body),(x2_body,y2_body),(255,0,0),thickness=2)
            # for tag in tags:
            #     ann_img = cv2.putText(ann_img,f"ID:{str(tag.tag_id)}",(x1_body,y2_body+30),cv2.FONT_HERSHEY_COMPLEX,1,color=(255,255,255),thickness=2,lineType=cv2.LINE_AA)
            #     break
            img_cropped_org = frame[y1_body:y2_body, x1_body:x2_body]
            height1, width1, _ = img_cropped_org.shape

            
            
            dataset = pose_model.cfg.data['test']['type']
            data_list = [
                {
                    'image_file': None,
                    'center': np.array([height/2, width/2], dtype=np.float32),
                    'scale': np.array([3.2062502, 4.275], dtype=np.float32),
                    'rotation': 0,
                    'bbox_score': 1,
                    'flip_pairs': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]],
                    'bbox_id': 0
                }
            ]
            
            target_width = 512
            target_height = 512
            img_cropped_resized , (top,bottom,left,right), (new_h,new_w,aspect_ratio)= resize_or_pad_image(img_cropped_org, (target_width, target_height))
            tensor_img = transform(img_cropped_resized)
            tensor_img_resized = resize_transform(tensor_img.unsqueeze(0)).to(device='cuda')
            x1,y1,x2,y2 = left, top , img_cropped_resized.shape[1] - right, img_cropped_resized.shape[0] - bottom
            person_results = [{'bbox': np.array([x1,y1,x2,y2])}]
            
            with torch.no_grad():
                output_layer_names = None
                dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
                if dataset_info is None:
                    warnings.warn(
                        'Please set `dataset_info` in the config.'
                        'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                        DeprecationWarning)
                else:
                    dataset_info = DatasetInfo(dataset_info)
                # cv2.imshow("Pose Input Image",img_cropped_resized)
                # cv2.waitKey(5000)
                pose_results, returned_outputs = inference_top_down_pose_model(
                                                    pose_model,
                                                    img_cropped_resized,
                                                    person_results,
                                                    format='xyxy',
                                                    dataset=dataset,
                                                    dataset_info=dataset_info,
                                                    return_heatmap=False,
                                                    outputs=None)
                keypoints = np.array(pose_results[0]['keypoints'])
                # print(keypoints)
                img_without_keypoints = img_cropped_resized.copy()
                keypoints_org = keypoints.copy()
                if new_h == img_cropped_org.shape[0] and new_w == img_cropped_org.shape[1]:
                    img_cropped_ann = draw_keypoints_and_skeleton(img_cropped_resized, keypoints[:, :2])
                    ann_img[y1_body:y2_body, x1_body:x2_body, :] = img_cropped_ann[y1:y2, x1:x2, :]
                elif new_w == target_width:  # Width was greater than target width, scaled to target width
                    scale = img_cropped_org.shape[1] / target_width
                    keypoints[:, :2] *= scale
                    img_cropped_resized = cv2.resize(img_cropped_resized, (int(img_cropped_resized.shape[1] * scale), int(img_cropped_resized.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                    # print(img_cropped_resized.shape)
                    img_cropped_ann = draw_keypoints_and_skeleton(img_cropped_resized, keypoints[:, :2])
                    height_body = y2_body - y1_body
                    y2_new = int(y2 * scale)
                    y1_new = int(y1 * scale)
                    if y2_new - y1_new != height_body:
                        y2_new = y1_new + height_body
                    ann_img[y1_body:y2_body, x1_body:x2_body, :] = img_cropped_ann[y1_new:y2_new, int(x1 * scale):int(x2 * scale), :]
                elif new_h == target_height:  # Height was greater than target height, scaled to target height
                    scale = img_cropped_org.shape[0] / target_height
                    # print("Cropped human Shape :",img_cropped_org.shape, " Scale: ",scale)
                    keypoints[:, :2] *= scale
                    img_cropped_resized = cv2.resize(img_cropped_resized, (int(img_cropped_resized.shape[1] * scale), int(img_cropped_resized.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                    # print("Image Resized Cropped shape : ",img_cropped_resized.shape)
                    img_cropped_ann = draw_keypoints_and_skeleton(img_cropped_resized, keypoints[:, :2])
                    width_body = x2_body - x1_body
                    x2_new = int(x2 * scale)
                    x1_new = int(x1 * scale)
                    if x2_new - x1_new != width_body:
                        x2_new = x1_new + width_body
                    ann_img[y1_body:y2_body, x1_body:x2_body, :] = img_cropped_ann[int(y1 * scale):int(y2 * scale), x1_new:x2_new, :]
                
                # cv2.imshow("pose",img_cropped_ann)
                # cv2.waitKey(500)
                # ann_img[y1:y2, x1:x2] = img_cropped_ann
                limbs = get_limbs(keypoints_org, img_without_keypoints, 0.0)
                for image in limbs:
                    cv2.imwrite(f"/home/uas-dtu/trauma_clone_for_testing/ViTPose/demo/gaussian_2/{image_number}.jpg",image)
                    image_number+=1

                limb_cropped_list = postprocess_limb(limbs)
                trauma_values = classify_list(limb_cropped_list, model_classify)
                ann_img = show_gaussian_on_image(limbs,ann_img,(x1_body,y1_body,x2_body,y2_body),trauma_values)

                out.write(ann_img)  
except Exception as e:
    print("Exception message:", e)
    print("\nFull exception traceback:")
    traceback.print_exc()
finally:
    # Ensure resources are properly released
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    