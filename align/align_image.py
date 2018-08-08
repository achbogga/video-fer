#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from align.detect_face import create_mtcnn, detect_face
from scipy import misc
from align.align_dataset_mtcnn import *
def initialize_mtcnn(gpu_memory_fraction, rect_minsize = 100, mtcnn_thresholds = [0.6, 0.7, 0.7], scale_factor = 0.709):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = create_mtcnn(sess, None)
    minsize = rect_minsize
    threshold = mtcnn_thresholds
    factor = scale_factor
    return pnet, rnet, onet, minsize, threshold, factor

def align_image(img, pnet, rnet, onet, minsize, threshold, factor, image_height = 160, image_width = 160, view_mtcnn_scores = False, NED_threshold = 0.3, filter_sideways = True, keep_aspect_ratio = False, discard_border_landmark_faces = False, margin = 32, visualize = False):
    #distance_info = {}
    if img.ndim<2:
        #create_dir(parent_dir+'/failed_only_two_channels')
        #img = cv2.imread(image_path)
        #cv2.imwrite(parent_dir+'/failed_only_two_channels/'+filename+'.png', img)
        #print('Unable to align "%s"' % image_path)
        return [], None, [], []
        
    elif img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:,:,0:3]
    
    bounding_boxes, landmark_points, final_info = detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    
    if not np.all(landmark_points):
        return [], None, [], []
    nrof_faces = bounding_boxes.shape[0]
    best_face_shape = (1,1,1)
    best_face_index = 0
    best_face_cropped = None
    best_face_bb = [0,0,0,0]
    best_face_conf = 0
    if nrof_faces>0:
        for i in range(nrof_faces):
            det = bounding_boxes[i, 0:4]
            conf = bounding_boxes[i, 4]
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                bounding_box_size = (det[2]-det[0])*(det[3]-det[1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[0]+det[2])/2-img_center[1], (det[1]+det[3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            if visualize:
                try:
                    draw_bb(landmark_points[:, i], bb, img, conf)
                except IndexError:
                    draw_bb(landmark_points[-len(landmark_points)%10:], bb, img, conf)
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            if (conf>best_face_conf):
                best_face_conf = conf
                best_face_index = i
                best_face_shape = cropped.shape
                best_face_cropped = cropped
                best_face_bb = bb
        bfi = best_face_index
        if view_mtcnn_scores:
            suffix = '-'+str(final_info[bfi]['Pnet_score'])+'-'+str(final_info[bfi]['Rnet_score'])+'-'+str(final_info[bfi]['Onet_score'])
            filename = filename.replace('_', suffix+'_')
            output_filename = output_class_dir+'/'+filename+'.png'

        if best_face_index>0 and np.all(landmark_points):
            landmarks = [landmark_points[i][best_face_index] for i in range(len(landmark_points))]
        elif len(landmark_points[0])>1 and np.all(landmark_points):
            landmarks = [landmark_points[i][0] for i in range(len(landmark_points))]
        elif np.all(landmark_points):
            landmarks = landmark_points
        #a function to be written to check if all the landmarks are in the bounding box to be cropped.
        l_flag = check_if_all_landmarks_in_img(landmarks, best_face_bb)
        if filter_sideways:
            sideways_flag = flag_sideways(best_face_bb, landmarks, NED_threshold)
            if sideways_flag:
                bb = best_face_bb
                best_face_cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                if keep_aspect_ratio:
                    scaled = resize_with_aspect_ratio(best_face_cropped, image_height)
                else:
                    scaled = misc.imresize(best_face_cropped, (image_height, image_width), interp='bilinear')
                #create_dir(parent_dir+'/_sideways_faces')
                #misc.imsave(parent_dir+'/_sideways_faces/'+filename+'.png', scaled)
                return [], None, [], []
        if not l_flag:
            bb = best_face_bb
            best_face_cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            if keep_aspect_ratio:
                scaled = resize_with_aspect_ratio(best_face_cropped, image_height)
            else:
                scaled = misc.imresize(best_face_cropped, (image_height, image_width), interp='bilinear')
            #create_dir(parent_dir+'/_failed_landmarks_out_of_bounding_box')
            #misc.imsave(parent_dir+'/_failed_landmarks_out_of_bounding_box/'+filename+'.png', scaled)
            return [], None, [], []
        info = get_distance_to_the_boundaries(landmarks, best_face_bb, img)
        if not flag_bad_landmarks(info) and discard_border_landmark_faces:
            bb = best_face_bb
            best_face_cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            if keep_aspect_ratio:
                scaled = resize_with_aspect_ratio(best_face_cropped, image_height)
            else:
                scaled = misc.imresize(best_face_cropped, (image_height, image_width), interp='bilinear')
            return [], None, [], []
        #distance_info[img] = [value[0] for value in list(info.values())]
        #distance_info[img].extend(list(img.shape))
        bb = best_face_bb
        best_face_cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if keep_aspect_ratio:
            scaled = resize_with_aspect_ratio(best_face_cropped, image_height)
        else:
            scaled = misc.imresize(best_face_cropped, (image_height, image_width), interp='bilinear')
        final_th = best_face_conf
        if best_face_conf>threshold[2]:
            if (not visualize):
                return scaled, final_th, bounding_boxes, landmark_points
            else:
                return img, final_th, bounding_boxes, landmark_points
        else:
            return [], None, [], []
    else:
        #print('Unable to align "%s"' % image_path)
        return [], None, [], []
