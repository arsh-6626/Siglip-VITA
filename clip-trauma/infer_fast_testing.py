# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
import time
import cv2
import numpy as np
from collections import deque
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

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

def Average(lst): 
    return sum(lst) / len(lst) 

def calculate_sigma(keypoints, k=0.25):
    sigma = np.ones([1, 3, 4], dtype=np.float16)
    for i in range(keypoints.shape[2]): 
            sigma[0, 0, i] = np.linalg.norm(keypoints[0, :2, i] - keypoints[1, :2, i]) * k * keypoints[0,3,i] * keypoints[1,3,i]
            sigma[0, 1, i] = np.linalg.norm(keypoints[1, :2, i] - keypoints[2, :2, i]) * k * keypoints[1,3,i] * keypoints[2,3,i]
            sigma[0, 2, i] = np.linalg.norm(keypoints[2, :2, i] - keypoints[0, :2, i]) * k * keypoints[2,3,i] * keypoints[0,3,i]
    return sigma

def gaussian(x, y, sigma):
    return np.exp(-(x**2 + y**2) / (2.0 * sigma**2))

def show_image(keypoints,image):
        img=image.copy()
        keypoints_int = np.array(keypoints,dtype=np.int16)
        for i in range(keypoints_int.shape[2]):
            limb = keypoints_int[:,:,i]
            img = cv2.line(img,pt1=(limb[0,0],limb[0,1]),pt2=(limb[1,0],limb[1,1]),color=(0,255,0),thickness=2)
            img = cv2.line(img,pt1=(limb[1,0],limb[1,1]),pt2=(limb[2,0],limb[2,1]),color=(0,255,0),thickness=2)
        return img

def apply_gaussian_splatting_image_fast(image,keypoints,sigmas,count,conf_thres):
    height, width = image.shape[:2]
    splatted_images = [np.zeros((height,width,3),dtype=np.float64),np.zeros((height,width,3),dtype=np.float64),np.zeros((height,width,3),dtype=np.float64),np.zeros((height,width,3),dtype=np.float64)]
    sigmas_mean = np.mean(sigmas,axis=1)
    radius = (2 * sigmas_mean).astype(np.int16)

    gaussian_weights_list = []

    for j in range(sigmas_mean.shape[1]):
        x = np.arange(-radius[0,j], radius[0,j] + 1)
        y = np.arange(-radius[0,j], radius[0,j] + 1)
        sigma = sigmas_mean[0,j]
        X, Y = np.meshgrid(x, y)
        gaussian_weights = gaussian(X, Y, sigma) / np.sum(gaussian(X, Y, sigma))
        gaussian_weights_list.append(gaussian_weights)

    # print(len(gaussian_weights_list))
    # keypoints = np.round(keypoints)
    midpoints = []
    
    # print(keypoints)
    for i in range(keypoints.shape[0] - 1):

        midpoint1 = (4 * keypoints[i, 0, :] + keypoints[i + 1, 0, :]) / 5, (4 * keypoints[i, 1, :] + keypoints[i + 1, 1, :]) / 5, \
                    (4 * keypoints[i, 2, :] + keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint1 = np.expand_dims(np.stack(midpoint1), axis=0)
        
        midpoint2 = (3 * keypoints[i, 0, :] + 2 * keypoints[i + 1, 0, :]) / 5, (3 * keypoints[i, 1, :] + 2 * keypoints[i + 1, 1, :]) / 5, \
                    (3 * keypoints[i, 2, :] + 2 * keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint2 = np.expand_dims(np.stack(midpoint2), axis=0)
        
        midpoint3 = (2 * keypoints[i, 0, :] + 3 * keypoints[i + 1, 0, :]) / 5, (2 * keypoints[i, 1, :] + 3 * keypoints[i + 1, 1, :]) / 5, \
                    (2 * keypoints[i, 2, :] + 3 * keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint3 = np.expand_dims(np.stack(midpoint3), axis=0)
        
        midpoint4 = (keypoints[i, 0, :] + 4 * keypoints[i + 1, 0, :]) / 5, (keypoints[i, 1, :] + 4 * keypoints[i + 1, 1, :]) / 5, \
                    ( keypoints[i, 2, :] + 4 * keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint4 = np.expand_dims(np.stack(midpoint4), axis=0)

        midpoints.append(np.concatenate((midpoint1, midpoint2, midpoint3, midpoint4), axis=0))

    midpoints = np.concatenate(midpoints, axis=0)  

    all_points = np.concatenate([keypoints,midpoints],axis=0)

    conf_score=all_points[:, -2, :]
    conf_mask=conf_score>=conf_thres
    all_points[:,-2,:]=conf_mask

    # last_two_columns = all_points[:, -2:, :]
    # bol=np.any(last_two_columns,axis=1)
    # combined_column = bol[:, np.newaxis, :]
    # all_points=all_points[:,:-2,:]
    # all_points=np.concatenate((all_points,combined_column),axis=1)

    calculation_matrix = np.zeros([all_points.shape[0], 8, all_points.shape[2]])
    calculation_matrix[:,0,:] = np.maximum(0, all_points[:,0,:] - radius)
    calculation_matrix[:,1,:] = np.minimum(width, all_points[:,0,:] + radius + 1)
    calculation_matrix[:,2,:] = np.maximum(0, all_points[:,1,:] - radius)
    calculation_matrix[:,3,:] = np.minimum(height, all_points[:,1,:] + radius + 1)
    calculation_matrix[:,4,:] = np.maximum(0, radius - all_points[:,0,:])
    calculation_matrix[:,5,:] = np.minimum(2 * radius + 1, radius - all_points[:,0,:] + width)
    calculation_matrix[:,6,:] = np.maximum(0, radius - all_points[:,1,:])
    calculation_matrix[:,7,:] = np.minimum(2 * radius + 1, radius - all_points[:,1,:] + height)

    calculation_matrix = calculation_matrix.astype(np.int16)

    # print(all_points.shape)
    saving_image_list = []
    saving_image_list.append(image)
    for i in range(all_points.shape[2]):
       
        # print(all_points)
        limb_points = all_points[:,:,i]
        # if limb_points[i][2]!=0:
        #      
        calculation_matrix_limb = calculation_matrix[:,:,i]
        splatted_image_limb = splatted_images[i]
        gaussian_weights_limb = gaussian_weights_list[i]
        # print(gaussian_weights_limb.shape)
        for j in range(limb_points.shape[0]):
            
            if limb_points[j,2] == 0 or limb_points[j,3]==0:
                continue

            # print(all_points)
            roi_image = image[calculation_matrix_limb[j,2]:calculation_matrix_limb[j,3],calculation_matrix_limb[j,0]:calculation_matrix_limb[j,1]]
            roi_splatted_image = splatted_image_limb[calculation_matrix_limb[j,2]:calculation_matrix_limb[j,3],calculation_matrix_limb[j,0]:calculation_matrix_limb[j,1]]
            roi_gaussian_weights = gaussian_weights_limb[calculation_matrix_limb[j,6]:calculation_matrix_limb[j,7], calculation_matrix_limb[j,4]:calculation_matrix_limb[j,5]]
            # print(roi_image.shape,roi_splatted_image.shape,roi_gaussian_weights.shape)

            if roi_gaussian_weights.shape[0] != roi_image.shape[0] or roi_gaussian_weights.shape[1] != roi_image.shape[1]:
                # # print(f"Shape mismatch before multiplication: roi_image: {roi_image.shape}, roi_gaussian_weights: {roi_gaussian_weights.shape}")
                min_height = min(roi_image.shape[0], roi_gaussian_weights.shape[0])
                min_width = min(roi_image.shape[1], roi_gaussian_weights.shape[1])
                roi_image = roi_image[:min_height, :min_width]
                roi_gaussian_weights = roi_gaussian_weights[:min_height, :min_width]
                roi_splatted_image = roi_splatted_image[:min_height, :min_width]
                # print(f"Shape after adjustment: roi_image: {roi_image.shape}, roi_gaussian_weights: {roi_gaussian_weights.shape}")

            if roi_image.ndim == 3:
                # print("Hello")
                roi_gaussian_weights = roi_gaussian_weights[..., np.newaxis]

            roi_splatted_image += roi_gaussian_weights * roi_image
            # splatted_image_limb[calculation_matrix_limb[j,2]:calculation_matrix_limb[j,3],calculation_matrix_limb[j,0]:calculation_matrix_limb[j,1]] = roi_splatted_image

        limb_image  = np.array(splatted_image_limb/np.max(splatted_image_limb)*255,dtype=np.uint8)
        # print("---")
        # print(np.unique(limb_image))
        red_threshold = 20
        blue_threshold = 20
        green_threshold = 20
        red_array = np.ones([height,width,1],np.float32) * red_threshold
        blue_array = np.ones([height,width,1],np.float32) * blue_threshold
        green_array = np.ones([height,width,1],np.float32) * green_threshold
        array = np.concatenate((blue_array,green_array,red_array),axis=2)
        mask = limb_image > array
        mask = np.expand_dims(np.all(mask,axis=2),axis=2)
        limb_image_new = mask * image

        cv2.imwrite(f"/infer_fast_{count}.jpg",limb_image)
        print(f"Saving Lmb Image : workshop_gaussian/infer_fast_{count}.jpg")
        count+=1

        saving_image_list.append(limb_image_new)
    # saving_image = np.concatenate(saving_image_list,axis=1)
    # cv2.imwrite(f"workshop_gaussian/infer_fast_{count}_full.jpg",saving_image)
    # print(f"Saving Combined Image : workshop_gaussian/infer_fast_{count}_full.jpg")
    # count+=1
    # print(f"Saving Image infer_fast_{count}.jpg")
    # splatted_image_normalized = splatted_image / np.max(splatted_image) * 255    
    
    return count

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    # parser.add_argument('pose_config', help='Config file for pose')
    # parser.add_argument('pose_checkpoint', help='Checkpoint file for pose',default='./home/gunmay/VitPose-s_RePoGen.pth')
    parser.add_argument('--video-path', type=str, help='Video path',default='/home/somin/output_vid2.mp4')
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='sample',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=5,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
       '../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py', 'vitpose_small.pth', device=args.device.lower())
    # print(pose_model)

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    i=0
    count =0
    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = 0
    while (cap.isOpened()):

        flag, img = cap.read()

        # img=cv2.resize(img,(256,192))
        if not flag:
            break

        frames+=1
        
        if frames % 2 == 0:
            continue

        # keep the person class bounding boxes.
        person_results = [{'bbox': np.array([0, 0, size[0], size[1]])}]

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        
        keypoints = np.array(pose_results[0]['keypoints'])

        # Left arm keypoints
        larm = np.expand_dims(np.concatenate((np.expand_dims(keypoints[5, :], axis=0), 
                                            np.expand_dims(keypoints[7, :], axis=0), 
                                            np.expand_dims(keypoints[9, :], axis=0)), axis=0), axis=2)

        # Right arm keypoints
        rarm = np.expand_dims(np.concatenate((np.expand_dims(keypoints[6, :], axis=0), 
                                            np.expand_dims(keypoints[8, :], axis=0), 
                                            np.expand_dims(keypoints[10, :], axis=0)), axis=0), axis=2)

        # Left leg keypoints
        lleg = np.expand_dims(np.concatenate((np.expand_dims(keypoints[11, :], axis=0), 
                                            np.expand_dims(keypoints[13, :], axis=0), 
                                            np.expand_dims(keypoints[15, :], axis=0)), axis=0), axis=2)

        # Right leg keypoints
        rleg = np.expand_dims(np.concatenate((np.expand_dims(keypoints[12, :], axis=0), 
                                            np.expand_dims(keypoints[14, :], axis=0), 
                                            np.expand_dims(keypoints[16, :], axis=0)), axis=0), axis=2)
        
        # Concatenating all four arrays along the channels (axis 0)
        bool_array = np.ones((3, 1, 4), dtype=bool)
        keypoints = np.concatenate((larm, rarm, lleg, rleg), axis=2)
        # print(keypoints)
        threshold = 1e-6
        
        # Get the indexes of elements that are 0 or very close to 0
        mask = np.abs(keypoints[:,:2,:]) >= threshold
        first_two_columns = mask[:, :2, :]

        # Combine the first two columns into a single column
        combined_column = np.any(first_two_columns, axis=1,keepdims=True)
        keypoints=np.concatenate((keypoints,combined_column),axis=1)

        del larm,rarm,lleg,rleg,bool_array
        # kimg=show_image(keypoints,"/home/sahil/images2/images2/frame_0.jpg")
        # splat_results = []
        # splat_results.append(kimg)
        sigmas = calculate_sigma(keypoints)
        count = apply_gaussian_splatting_image_fast(img,keypoints,sigmas,count,conf_thres=0.5)
        # cv2.imwrite(splatted_filename, final)
        # print(f"Image saved splatted_{i}.jpg")
        # i+=1
        count +=1
        
    end_time = time.time()
    print(f"Total Time Taken : {end_time - start_time}")
    print(f"FPS : {frames/(end_time - start_time)}")

def apply_gaussian_splatting_image_pipeline(image,keypoints,sigmas,conf_thres):
    height, width = image.shape[:2]
    splatted_images = [np.zeros((height,width,3),dtype=np.float64),np.zeros((height,width,3),dtype=np.float64),np.zeros((height,width,3),dtype=np.float64),np.zeros((height,width,3),dtype=np.float64)]
    sigmas_mean = np.mean(sigmas,axis=1)
    radius = (2 * sigmas_mean).astype(np.int16)

    gaussian_weights_list = []

    for j in range(sigmas_mean.shape[1]):
        x = np.arange(-radius[0,j], radius[0,j] + 1)
        y = np.arange(-radius[0,j], radius[0,j] + 1)
        sigma = sigmas_mean[0,j]
        X, Y = np.meshgrid(x, y)
        gaussian_weights = gaussian(X, Y, sigma) / np.sum(gaussian(X, Y, sigma))
        gaussian_weights_list.append(gaussian_weights)

    # print(len(gaussian_weights_list))
    # keypoints = np.round(keypoints)
    midpoints = []
    
    # print(keypoints)
    for i in range(keypoints.shape[0] - 1):

        midpoint1 = (4 * keypoints[i, 0, :] + keypoints[i + 1, 0, :]) / 5, (4 * keypoints[i, 1, :] + keypoints[i + 1, 1, :]) / 5, \
                    (4 * keypoints[i, 2, :] + keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint1 = np.expand_dims(np.stack(midpoint1), axis=0)
        
        midpoint2 = (3 * keypoints[i, 0, :] + 2 * keypoints[i + 1, 0, :]) / 5, (3 * keypoints[i, 1, :] + 2 * keypoints[i + 1, 1, :]) / 5, \
                    (3 * keypoints[i, 2, :] + 2 * keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint2 = np.expand_dims(np.stack(midpoint2), axis=0)
        
        midpoint3 = (2 * keypoints[i, 0, :] + 3 * keypoints[i + 1, 0, :]) / 5, (2 * keypoints[i, 1, :] + 3 * keypoints[i + 1, 1, :]) / 5, \
                    (2 * keypoints[i, 2, :] + 3 * keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint3 = np.expand_dims(np.stack(midpoint3), axis=0)
        
        midpoint4 = (keypoints[i, 0, :] + 4 * keypoints[i + 1, 0, :]) / 5, (keypoints[i, 1, :] + 4 * keypoints[i + 1, 1, :]) / 5, \
                    ( keypoints[i, 2, :] + 4 * keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint4 = np.expand_dims(np.stack(midpoint4), axis=0)

        midpoints.append(np.concatenate((midpoint1, midpoint2, midpoint3, midpoint4), axis=0))

    midpoints = np.concatenate(midpoints, axis=0)  
    print(midpoints.shape)
    midpoints = np.zeros_like(midpoints)
    
    all_points = np.concatenate([keypoints,midpoints],axis=0)

    conf_score=all_points[:, -2, :]
    conf_mask=conf_score>=conf_thres
    all_points[:,-2,:]=conf_mask

    # last_two_columns = all_points[:, -2:, :]
    # bol=np.any(last_two_columns,axis=1)
    # combined_column = bol[:, np.newaxis, :]
    # all_points=all_points[:,:-2,:]
    # all_points=np.concatenate((all_points,combined_column),axis=1)

    calculation_matrix = np.zeros([all_points.shape[0], 8, all_points.shape[2]])
    calculation_matrix[:,0,:] = np.maximum(0, all_points[:,0,:] - radius)
    calculation_matrix[:,1,:] = np.minimum(width, all_points[:,0,:] + radius + 1)
    calculation_matrix[:,2,:] = np.maximum(0, all_points[:,1,:] - radius)
    calculation_matrix[:,3,:] = np.minimum(height, all_points[:,1,:] + radius + 1)
    calculation_matrix[:,4,:] = np.maximum(0, radius - all_points[:,0,:])
    calculation_matrix[:,5,:] = np.minimum(2 * radius + 1, radius - all_points[:,0,:] + width)
    calculation_matrix[:,6,:] = np.maximum(0, radius - all_points[:,1,:])
    calculation_matrix[:,7,:] = np.minimum(2 * radius + 1, radius - all_points[:,1,:] + height)

    calculation_matrix = calculation_matrix.astype(np.int16)

    # print(all_points.shape)
    saving_image_list = []
    for i in range(all_points.shape[2]):
       
        # print(all_points)
        limb_points = all_points[:,:,i]
        # if limb_points[i][2]!=0:
        #      
        calculation_matrix_limb = calculation_matrix[:,:,i]
        splatted_image_limb = splatted_images[i]
        gaussian_weights_limb = gaussian_weights_list[i]
        # print(gaussian_weights_limb.shape)
        for j in range(limb_points.shape[0]):
            
            if limb_points[j,2] == 0 or limb_points[j,3]==0:
                continue

            # print(all_points)
            roi_image = image[calculation_matrix_limb[j,2]:calculation_matrix_limb[j,3],calculation_matrix_limb[j,0]:calculation_matrix_limb[j,1]]
            roi_splatted_image = splatted_image_limb[calculation_matrix_limb[j,2]:calculation_matrix_limb[j,3],calculation_matrix_limb[j,0]:calculation_matrix_limb[j,1]]
            roi_gaussian_weights = gaussian_weights_limb[calculation_matrix_limb[j,6]:calculation_matrix_limb[j,7], calculation_matrix_limb[j,4]:calculation_matrix_limb[j,5]]
            # print(roi_image.shape,roi_splatted_image.shape,roi_gaussian_weights.shape)

            if roi_gaussian_weights.shape[0] != roi_image.shape[0] or roi_gaussian_weights.shape[1] != roi_image.shape[1]:
                # # print(f"Shape mismatch before multiplication: roi_image: {roi_image.shape}, roi_gaussian_weights: {roi_gaussian_weights.shape}")
                min_height = min(roi_image.shape[0], roi_gaussian_weights.shape[0])
                min_width = min(roi_image.shape[1], roi_gaussian_weights.shape[1])
                roi_image = roi_image[:min_height, :min_width]
                roi_gaussian_weights = roi_gaussian_weights[:min_height, :min_width]
                roi_splatted_image = roi_splatted_image[:min_height, :min_width]
                # print(f"Shape after adjustment: roi_image: {roi_image.shape}, roi_gaussian_weights: {roi_gaussian_weights.shape}")

            if roi_image.ndim == 3:
                # print("Hello")
                roi_gaussian_weights = roi_gaussian_weights[..., np.newaxis]

            roi_splatted_image += roi_gaussian_weights * roi_image
            # splatted_image_limb[calculation_matrix_limb[j,2]:calculation_matrix_limb[j,3],calculation_matrix_limb[j,0]:calculation_matrix_limb[j,1]] = roi_splatted_image

        if np.array_equal(splatted_image_limb, np.zeros((height, width, 3), dtype=np.float64)):
            limb_image = np.array(splatted_image_limb,dtype=np.uint8)
        else:
            limb_image  = np.array(splatted_image_limb/np.max(splatted_image_limb)*255,dtype=np.uint8)
        saving_image_list.append(limb_image)
    
    return saving_image_list

def get_limbs(keypoints,img,conf_thres):
        
    larm = np.expand_dims(np.concatenate((np.expand_dims(keypoints[5, :], axis=0), 
                                        np.expand_dims(keypoints[7, :], axis=0), 
                                        np.expand_dims(keypoints[9, :], axis=0)), axis=0), axis=2)

    # Right arm keypoints
    rarm = np.expand_dims(np.concatenate((np.expand_dims(keypoints[6, :], axis=0), 
                                        np.expand_dims(keypoints[8, :], axis=0), 
                                        np.expand_dims(keypoints[10, :], axis=0)), axis=0), axis=2)

    # Left leg keypoints
    lleg = np.expand_dims(np.concatenate((np.expand_dims(keypoints[11, :], axis=0), 
                                        np.expand_dims(keypoints[13, :], axis=0), 
                                        np.expand_dims(keypoints[15, :], axis=0)), axis=0), axis=2)

    # Right leg keypoints
    rleg = np.expand_dims(np.concatenate((np.expand_dims(keypoints[12, :], axis=0), 
                                        np.expand_dims(keypoints[14, :], axis=0), 
                                        np.expand_dims(keypoints[16, :], axis=0)), axis=0), axis=2)
    
    # Concatenating all four arrays along the channels (axis 0)
    bool_array = np.ones((3, 1, 4), dtype=bool)
    keypoints = np.concatenate((larm, rarm, lleg, rleg), axis=2)
    # print(keypoints)
    threshold = 1e-6
    
    # Get the indexes of elements that are 0 or very close to 0
    mask = np.abs(keypoints[:,:2,:]) >= threshold
    first_two_columns = mask[:, :2, :]

    # Combine the first two columns into a single column
    combined_column = np.any(first_two_columns, axis=1,keepdims=True)
    keypoints=np.concatenate((keypoints,combined_column),axis=1)

    del larm,rarm,lleg,rleg,bool_array,combined_column,first_two_columns,mask,threshold
    sigmas = calculate_sigma(keypoints)
    limb_images_list = apply_gaussian_splatting_image_pipeline(img,keypoints,sigmas,conf_thres)
    return limb_images_list

def get_cropped_limbs():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    # parser.add_argument('pose_config', help='Config file for pose')
    # parser.add_argument('pose_checkpoint', help='Checkpoint file for pose',default='./home/gunmay/VitPose-s_RePoGen.pth')
    parser.add_argument('--video-path', type=str, help='Video path',default='/home/somin/output_vid.mp4')
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='sample',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=5,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
       '../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py', 'vitpose_small.pth', device=args.device.lower())
    # print(pose_model)

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    i=0
    count =0
    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = 0
    while (cap.isOpened()):

        flag, img = cap.read()

        # img=cv2.resize(img,(256,192))
        if not flag:
            break

        frames+=1
        
        if frames % 2 == 0:
            continue

        # keep the person class bounding boxes.
        person_results = [{'bbox': np.array([0, 0, size[0], size[1]])}]

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        
        keypoints = np.array(pose_results[0]['keypoints'])

        # Left arm keypoints
        larm = np.expand_dims(np.concatenate((np.expand_dims(keypoints[5, :], axis=0), 
                                            np.expand_dims(keypoints[7, :], axis=0), 
                                            np.expand_dims(keypoints[9, :], axis=0)), axis=0), axis=2)

        # Right arm keypoints
        rarm = np.expand_dims(np.concatenate((np.expand_dims(keypoints[6, :], axis=0), 
                                            np.expand_dims(keypoints[8, :], axis=0), 
                                            np.expand_dims(keypoints[10, :], axis=0)), axis=0), axis=2)

        # Left leg keypoints
        lleg = np.expand_dims(np.concatenate((np.expand_dims(keypoints[11, :], axis=0), 
                                            np.expand_dims(keypoints[13, :], axis=0), 
                                            np.expand_dims(keypoints[15, :], axis=0)), axis=0), axis=2)

        # Right leg keypoints
        rleg = np.expand_dims(np.concatenate((np.expand_dims(keypoints[12, :], axis=0), 
                                            np.expand_dims(keypoints[14, :], axis=0), 
                                            np.expand_dims(keypoints[16, :], axis=0)), axis=0), axis=2)
        
        # Concatenating all four arrays along the channels (axis 0)
        bool_array = np.ones((3, 1, 4), dtype=bool)
        keypoints = np.concatenate((larm, rarm, lleg, rleg), axis=2)
        # print(keypoints)
        threshold = 1e-6
        
        # Get the indexes of elements that are 0 or very close to 0
        mask = np.abs(keypoints[:,:2,:]) >= threshold
        first_two_columns = mask[:, :2, :]

        # Combine the first two columns into a single column
        combined_column = np.any(first_two_columns, axis=1,keepdims=True)
        keypoints=np.concatenate((keypoints,combined_column),axis=1)

        del larm,rarm,lleg,rleg,bool_array
        # kimg=show_image(keypoints,"/home/sahil/images2/images2/frame_0.jpg")
        # splat_results = []
        # splat_results.append(kimg)
        sigmas = calculate_sigma(keypoints)
        count = gaussian_and_bfs(img,keypoints,sigmas,count,conf_thres=0.5,output_folder="gaussian_and_bfs")
    end_time = time.time()
    print(f"Total Time Taken : {end_time - start_time}")
    print(f"FPS : {frames/(end_time - start_time)}")

def gaussian_and_bfs(image,keypoints,sigmas,count,conf_thres,output_folder):
    height, width = image.shape[:2]
    splatted_images = [np.zeros((height,width,3),dtype=np.float64),np.zeros((height,width,3),dtype=np.float64),np.zeros((height,width,3),dtype=np.float64),np.zeros((height,width,3),dtype=np.float64)]
    sigmas_mean = np.mean(sigmas,axis=1)
    radius = (2 * sigmas_mean).astype(np.int16)

    gaussian_weights_list = []

    for j in range(sigmas_mean.shape[1]):
        x = np.arange(-radius[0,j], radius[0,j] + 1)
        y = np.arange(-radius[0,j], radius[0,j] + 1)
        sigma = sigmas_mean[0,j]
        X, Y = np.meshgrid(x, y)
        gaussian_weights = gaussian(X, Y, sigma) / np.sum(gaussian(X, Y, sigma))
        gaussian_weights_list.append(gaussian_weights)

    # print(len(gaussian_weights_list))
    # keypoints = np.round(keypoints)
    midpoints = []
    
    # print(keypoints)
    for i in range(keypoints.shape[0] - 1):

        midpoint1 = (4 * keypoints[i, 0, :] + keypoints[i + 1, 0, :]) / 5, (4 * keypoints[i, 1, :] + keypoints[i + 1, 1, :]) / 5, \
                    (4 * keypoints[i, 2, :] + keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint1 = np.expand_dims(np.stack(midpoint1), axis=0)
        
        midpoint2 = (3 * keypoints[i, 0, :] + 2 * keypoints[i + 1, 0, :]) / 5, (3 * keypoints[i, 1, :] + 2 * keypoints[i + 1, 1, :]) / 5, \
                    (3 * keypoints[i, 2, :] + 2 * keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint2 = np.expand_dims(np.stack(midpoint2), axis=0)
        
        midpoint3 = (2 * keypoints[i, 0, :] + 3 * keypoints[i + 1, 0, :]) / 5, (2 * keypoints[i, 1, :] + 3 * keypoints[i + 1, 1, :]) / 5, \
                    (2 * keypoints[i, 2, :] + 3 * keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint3 = np.expand_dims(np.stack(midpoint3), axis=0)
        
        midpoint4 = (keypoints[i, 0, :] + 4 * keypoints[i + 1, 0, :]) / 5, (keypoints[i, 1, :] + 4 * keypoints[i + 1, 1, :]) / 5, \
                    ( keypoints[i, 2, :] + 4 * keypoints[i + 1, 2, :]) / 5, keypoints[i,3,:] * keypoints[i+1,3,:]
        midpoint4 = np.expand_dims(np.stack(midpoint4), axis=0)

        midpoints.append(np.concatenate((midpoint1, midpoint2, midpoint3, midpoint4), axis=0))

    midpoints = np.concatenate(midpoints, axis=0)  

    all_points = np.concatenate([keypoints,midpoints],axis=0)

    conf_score=all_points[:, -2, :]
    conf_mask=conf_score>=conf_thres
    all_points[:,-2,:]=conf_mask

    # last_two_columns = all_points[:, -2:, :]
    # bol=np.any(last_two_columns,axis=1)
    # combined_column = bol[:, np.newaxis, :]
    # all_points=all_points[:,:-2,:]
    # all_points=np.concatenate((all_points,combined_column),axis=1)

    calculation_matrix = np.zeros([all_points.shape[0], 8, all_points.shape[2]])
    calculation_matrix[:,0,:] = np.maximum(0, all_points[:,0,:] - radius)
    calculation_matrix[:,1,:] = np.minimum(width, all_points[:,0,:] + radius + 1)
    calculation_matrix[:,2,:] = np.maximum(0, all_points[:,1,:] - radius)
    calculation_matrix[:,3,:] = np.minimum(height, all_points[:,1,:] + radius + 1)
    calculation_matrix[:,4,:] = np.maximum(0, radius - all_points[:,0,:])
    calculation_matrix[:,5,:] = np.minimum(2 * radius + 1, radius - all_points[:,0,:] + width)
    calculation_matrix[:,6,:] = np.maximum(0, radius - all_points[:,1,:])
    calculation_matrix[:,7,:] = np.minimum(2 * radius + 1, radius - all_points[:,1,:] + height)
    calculation_matrix = calculation_matrix.astype(np.int16)

    # print(all_points.shape)
    saving_image_list = []
    for i in range(all_points.shape[2]):
       
        # print(all_points)
        limb_points = all_points[:,:,i]
        # if limb_points[i][2]!=0:
        #      
        calculation_matrix_limb = calculation_matrix[:,:,i]
        splatted_image_limb = splatted_images[i]
        gaussian_weights_limb = gaussian_weights_list[i]
        # print(gaussian_weights_limb.shape)
        for j in range(limb_points.shape[0]):
            
            if limb_points[j,2] == 0 or limb_points[j,3]==0:
                continue

            # print(all_points)
            roi_image = image[calculation_matrix_limb[j,2]:calculation_matrix_limb[j,3],calculation_matrix_limb[j,0]:calculation_matrix_limb[j,1]]
            roi_splatted_image = splatted_image_limb[calculation_matrix_limb[j,2]:calculation_matrix_limb[j,3],calculation_matrix_limb[j,0]:calculation_matrix_limb[j,1]]
            roi_gaussian_weights = gaussian_weights_limb[calculation_matrix_limb[j,6]:calculation_matrix_limb[j,7], calculation_matrix_limb[j,4]:calculation_matrix_limb[j,5]]
            # print(roi_image.shape,roi_splatted_image.shape,roi_gaussian_weights.shape)

            if roi_gaussian_weights.shape[0] != roi_image.shape[0] or roi_gaussian_weights.shape[1] != roi_image.shape[1]:
                # # print(f"Shape mismatch before multiplication: roi_image: {roi_image.shape}, roi_gaussian_weights: {roi_gaussian_weights.shape}")
                min_height = min(roi_image.shape[0], roi_gaussian_weights.shape[0])
                min_width = min(roi_image.shape[1], roi_gaussian_weights.shape[1])
                roi_image = roi_image[:min_height, :min_width]
                roi_gaussian_weights = roi_gaussian_weights[:min_height, :min_width]
                roi_splatted_image = roi_splatted_image[:min_height, :min_width]
                # print(f"Shape after adjustment: roi_image: {roi_image.shape}, roi_gaussian_weights: {roi_gaussian_weights.shape}")

            if roi_image.ndim == 3:
                # print("Hello")
                roi_gaussian_weights = roi_gaussian_weights[..., np.newaxis]

            roi_splatted_image += roi_gaussian_weights * roi_image
            # splatted_image_limb[calculation_matrix_limb[j,2]:calculation_matrix_limb[j,3],calculation_matrix_limb[j,0]:calculation_matrix_limb[j,1]] = roi_splatted_image

        if np.array_equal(splatted_image_limb, np.zeros((height, width, 3), dtype=np.float64)):
            limb_image = np.array(splatted_image_limb,dtype=np.uint8)
        else:
            limb_image  = np.array(splatted_image_limb/np.max(splatted_image_limb)*255,dtype=np.uint8)
        
        img_resized = cv2.cvtColor(cv2.resize(limb_image, (limb_image.shape[1]//8, limb_image.shape[0]//8)), cv2.COLOR_BGR2GRAY)
        # print(img_resized.shape)
        visited= np.zeros(img_resized.shape, dtype=bool)
        max_x, max_y, min_x, min_y = bfs(img_resized ,0,0, visited)
        image_cropped = cv2.resize(limb_image[8*min_x:8*max_x,8*min_y:8*max_y], (256,256))
        
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        cv2.imwrite(f"{output_folder}/frame_{count}.png",image_cropped)
        print(f"Saving image {output_folder}/frame_{count}.png")
        count +=1
    
    return count

    
if __name__ == '__main__':
    get_cropped_limbs()
