import torch
import requests
import numpy as np
import cv2
import os
from ultralytics import YOLO
from collections import deque
from PIL import Image
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

class PoseProcessor:
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.person_model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365", 
            device_map=device
        )
        self.pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-base-simple", 
            device_map=device
        )
        self.count = 0
        self.count_keypoint = 0
        
    def detect_persons(image, threshold=0.3)
        model = YOLO('./best_body.pt')
        if hasattr(image, 'convert'):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = model(image)[0]
        person_boxes = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if int(cls.item()) == 0 and conf.item() >= threshold:
                x1, y1, x2, y2 = box.cpu().numpy()
                person_boxes.append([x1, y1, x2 - x1, y2 - y1])
        return None if not person_boxes else np.array(person_boxes)

    
    def estimate_pose(self, image, person_boxes, threshold=0.3):
        inputs = self.pose_processor(
            image, 
            boxes=[person_boxes], 
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.pose_model(**inputs)   
        pose_results = self.pose_processor.post_process_pose_estimation(
            outputs, 
            boxes=[person_boxes], 
            threshold=threshold
        )  
        return pose_results[0]
    
    
    def extract_limb_keypoints(self, keypoints, scores):
        left_shoulder, right_shoulder = 5, 6
        left_elbow, right_elbow = 7, 8
        left_wrist, right_wrist = 9, 10
        left_hip, right_hip = 11, 12
        left_knee, right_knee = 13, 14
        left_ankle, right_ankle = 15, 16
        limbs = {
            'larm': [left_shoulder, left_elbow, left_wrist],
            'rarm': [right_shoulder, right_elbow, right_wrist],
            'lleg': [left_hip, left_knee, left_ankle],
            'rleg': [right_hip, right_knee, right_ankle]
        }
        
        limb_keypoints = []
        for limb_name, indices in limbs.items():
            limb_points = []
            for idx in indices:
                if idx < len(keypoints):
                    x, y = keypoints[idx]
                    score = scores[idx] if idx < len(scores) else 0.0
                    conf = 1.0 if score > 0.3 else 0.0
                    limb_points.append([x.item(), y.item(), score.item(), conf])
                else:
                    limb_points.append([0.0, 0.0, 0.0, 0.0])
            limb_keypoints.append(limb_points)
        limb_keypoints = np.array(limb_keypoints)
        limb_keypoints = np.transpose(limb_keypoints, (1, 2, 0))
        
        return limb_keypoints
    
    def gaussian(self, x, y, sigma):
        return np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    
    def calculate_sigma(self, keypoints, k=0.24):
        sigma = np.ones([1, 3, 4], dtype=np.float16)
        for i in range(keypoints.shape[2]): 
            sigma[0, 0, i] = np.linalg.norm(keypoints[0, :2, i] - keypoints[1, :2, i]) * k * keypoints[0,3,i] * keypoints[1,3,i]
            sigma[0, 1, i] = np.linalg.norm(keypoints[1, :2, i] - keypoints[2, :2, i]) * k * keypoints[1,3,i] * keypoints[2,3,i]
            sigma[0, 2, i] = np.linalg.norm(keypoints[2, :2, i] - keypoints[0, :2, i]) * k * keypoints[2,3,i] * keypoints[0,3,i]
        return sigma
    
    def bfs(self, image, startrow, startcol, visited):
        q = deque()
        row, col = image.shape
        max_x, min_x, max_y, min_y = 0, row, 0, col 
        
        visited[startrow][startcol] = True
        q.append([startrow, startcol])
        
        while q:
            currentnode = q.popleft()
            currentrow, currentcol = currentnode
            color = image[currentrow, currentcol]
            
            if color != 0:
                if currentrow > max_x:
                    max_x = currentrow
                if currentrow < min_x:
                    min_x = currentrow
                if currentcol > max_y:
                    max_y = currentcol
                if currentcol < min_y:
                    min_y = currentcol
            
            for i in range(-1, 2):
                for j in range(-1, 2):
                    new_row = currentrow - i
                    new_col = currentcol - j
                    if (0 <= new_row < row and 0 <= new_col < col and 
                        not visited[new_row][new_col]):
                        visited[new_row][new_col] = True
                        q.append([new_row, new_col])
        
        return max_x, max_y, min_x, min_y
    
    def get_gaussian_and_bfs_limbs(self, image, keypoints, sigmas, conf_thres=0.0):
        limb_images_list = []
        height, width = image.shape[:2]
        splatted_images = [np.zeros((height, width, 3), dtype=np.float64) for _ in range(4)]
        sigmas_mean = np.mean(sigmas, axis=1)
        radius = (2 * sigmas_mean).astype(np.int16)
        
        gaussian_weights_list = []
        
        for j in range(sigmas_mean.shape[1]):
            x = np.arange(-radius[0,j], radius[0,j] + 1)
            y = np.arange(-radius[0,j], radius[0,j] + 1)
            sigma = sigmas_mean[0,j]
            X, Y = np.meshgrid(x, y)
            gaussian_weights = self.gaussian(X, Y, sigma) / np.sum(self.gaussian(X, Y, sigma))
            gaussian_weights_list.append(gaussian_weights)
    
        midpoints = []
        for i in range(keypoints.shape[0] - 1):
            for ratio in [(4, 1), (3, 2), (2, 3), (1, 4)]:
                r1, r2 = ratio
                midpoint = (
                    (r1 * keypoints[i, 0, :] + r2 * keypoints[i + 1, 0, :]) / 5,
                    (r1 * keypoints[i, 1, :] + r2 * keypoints[i + 1, 1, :]) / 5,
                    (r1 * keypoints[i, 2, :] + r2 * keypoints[i + 1, 2, :]) / 5,
                    keypoints[i, 3, :] * keypoints[i + 1, 3, :]
                )
                midpoints.append(np.expand_dims(np.stack(midpoint), axis=0))
        
        if midpoints:
            midpoints = np.concatenate(midpoints, axis=0)
            all_points = np.concatenate([keypoints, midpoints], axis=0)
        else:
            all_points = keypoints
        conf_score = all_points[:, -2, :]
        conf_mask = conf_score >= conf_thres
        all_points[:, -2, :] = conf_mask
        calculation_matrix = np.zeros([all_points.shape[0], 8, all_points.shape[2]])
        calculation_matrix[:, 0, :] = np.maximum(0, all_points[:, 0, :] - radius)
        calculation_matrix[:, 1, :] = np.minimum(width, all_points[:, 0, :] + radius + 1)
        calculation_matrix[:, 2, :] = np.maximum(0, all_points[:, 1, :] - radius)
        calculation_matrix[:, 3, :] = np.minimum(height, all_points[:, 1, :] + radius + 1)
        calculation_matrix[:, 4, :] = np.maximum(0, radius - all_points[:, 0, :])
        calculation_matrix[:, 5, :] = np.minimum(2 * radius + 1, radius - all_points[:, 0, :] + width)
        calculation_matrix[:, 6, :] = np.maximum(0, radius - all_points[:, 1, :])
        calculation_matrix[:, 7, :] = np.minimum(2 * radius + 1, radius - all_points[:, 1, :] + height)
        calculation_matrix = calculation_matrix.astype(np.int16)
        for i in range(all_points.shape[2]):
            limb_points = all_points[:, :, i]
            calculation_matrix_limb = calculation_matrix[:, :, i]
            splatted_image_limb = splatted_images[i]
            gaussian_weights_limb = gaussian_weights_list[i]
            
            for j in range(limb_points.shape[0]):
                if limb_points[j, 2] == 0 or limb_points[j, 3] == 0:
                    continue
                
                roi_image = image[
                    calculation_matrix_limb[j, 2]:calculation_matrix_limb[j, 3],
                    calculation_matrix_limb[j, 0]:calculation_matrix_limb[j, 1]
                ]
                roi_splatted_image = splatted_image_limb[
                    calculation_matrix_limb[j, 2]:calculation_matrix_limb[j, 3],
                    calculation_matrix_limb[j, 0]:calculation_matrix_limb[j, 1]
                ]
                roi_gaussian_weights = gaussian_weights_limb[
                    calculation_matrix_limb[j, 6]:calculation_matrix_limb[j, 7], 
                    calculation_matrix_limb[j, 4]:calculation_matrix_limb[j, 5]
                ]
                
                # Handle shape mismatches
                if (roi_gaussian_weights.shape[0] != roi_image.shape[0] or 
                    roi_gaussian_weights.shape[1] != roi_image.shape[1]):
                    min_height = min(roi_image.shape[0], roi_gaussian_weights.shape[0])
                    min_width = min(roi_image.shape[1], roi_gaussian_weights.shape[1])
                    roi_image = roi_image[:min_height, :min_width]
                    roi_gaussian_weights = roi_gaussian_weights[:min_height, :min_width]
                    roi_splatted_image = roi_splatted_image[:min_height, :min_width]
                
                if roi_image.ndim == 3:
                    roi_gaussian_weights = roi_gaussian_weights[..., np.newaxis]
                
                roi_splatted_image += roi_gaussian_weights * roi_image
            if np.array_equal(splatted_image_limb, np.zeros((height, width, 3), dtype=np.float64)):
                limb_image = np.array(splatted_image_limb, dtype=np.uint8)
            else:
                limb_image = np.array(splatted_image_limb / np.max(splatted_image_limb) * 255, dtype=np.uint8)
            
            try:
                img_resized = cv2.cvtColor(
                    cv2.resize(limb_image, (limb_image.shape[1]//8, limb_image.shape[0]//8)), 
                    cv2.COLOR_BGR2GRAY
                )

                visited = np.zeros(img_resized.shape, dtype=bool)
                max_x, max_y, min_x, min_y = self.bfs(img_resized, 0, 0, visited)
                image_cropped = cv2.resize(
                    limb_image[8*min_x:8*max_x, 8*min_y:8*max_y], 
                    (256, 256)
                )
                limb_images_list.append(image_cropped)
            except:
                continue
        return limb_images_list
    
    def process_image(self, image_path, output_folder):
        if isinstance(image_path, str):
            if image_path.startswith('http'):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
        else:
            image = image_path
        image_np = np.array(image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        person_boxes = self.detect_persons(image)
        if person_boxes is None:
            print("No persons detected")
            return None
        areas = person_boxes[:, 2] * person_boxes[:, 3]
        max_area_idx = np.argmax(areas)
        person_box = person_boxes[max_area_idx:max_area_idx+1]
        pose_results = self.estimate_pose(image, person_box)
        if not pose_results:
            print("No pose detected")
            return None
        person_pose = pose_results[0]
        keypoints = person_pose["keypoints"]
        scores = person_pose["scores"]
        face_region = self.extract_face_region(image_np, keypoints, scores)
        chest_region = self.extract_chest_region(image_np, keypoints, scores)
        limb_keypoints = self.extract_limb_keypoints(keypoints, scores)
        sigmas = self.calculate_sigma(limb_keypoints)
        limb_images = self.get_gaussian_and_bfs_limbs(image_np, limb_keypoints, sigmas)
        self.save_results(face_region, chest_region, limb_images, output_folder)
        
        return {
            'face': face_region,
            'chest': chest_region,
            'limbs': limb_images,
            'keypoints': keypoints,
            'scores': scores
        }
    
    def save_results(self, face_region, chest_region, limb_images, output_folder):
        self.count_keypoint += 1        
        limb_folders = ['face', 'chest', 'larm', 'rarm', 'lleg', 'rleg']
        for limb in limb_folders:
            os.makedirs(os.path.join(output_folder, limb), exist_ok=True)
        if face_region is not None and face_region.size > 0:
            cv2.imwrite(
                os.path.join(output_folder, 'face', f"face_{self.count_keypoint}.png"), 
                face_region
            )
        if chest_region is not None and chest_region.size > 0:
            cv2.imwrite(
                os.path.join(output_folder, 'chest', f"chest_{self.count_keypoint}.png"), 
                chest_region
            )
        limb_names = ['larm', 'rarm', 'lleg', 'rleg']
        for i, limb_name in enumerate(limb_names):
            if i < len(limb_images):
                cv2.imwrite(
                    os.path.join(output_folder, limb_name, f"{limb_name}_{self.count_keypoint}.png"), 
                    limb_images[i]
                )


if __name__ == "__main__":
    processor = PoseProcessor()
    img_folder = "/home/cha0s/Downloads/c_data-20250601T135540Z-1-001/c_data"
    output_folder = "./test"
    call_count = 0    
    for img_name in sorted(os.listdir(img_folder)):
        img_path = os.path.join(img_folder, img_name)
        try:
            result = processor.process_image(img_path, output_folder)
            call_count += 1
            print(f"Processed {img_name} - {call_count}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    print(f"Total processed: {call_count}")