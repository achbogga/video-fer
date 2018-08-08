#!/usr/bin/env python
"""Performs face alignment and stores face thumbnails in the output directory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face_ovi
import random
import math
import json
import cv2
from time import sleep
from numpy.linalg import norm
from PIL import Image, ImageOps
from scipy.spatial.distance import euclidean

def intra_point_distance(points):
    sum_of_distances_to_nose = euclidean(points['left_eye'], points['nose']) + euclidean(points['right_eye'], points['nose']) + euclidean(points['left_mouth'], points['nose']) + euclidean(points['right_mouth'], points['nose'])
    return sum_of_distances_to_nose

def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def flag_sideways(bb, landmarks, NED_threshold):
    top_left = (bb[0], bb[1])
    top_right = (bb[2], bb[1])
    bottom_left = (bb[0], bb[3])
    bottom_right = (bb[2], bb[3])
    upper_boundary = [top_left, top_right]
    left_boundary = [top_left, bottom_left]
    right_boundary = [top_right, bottom_right]
    bottom_boundary = [bottom_left, bottom_right]

    n = int(len(landmarks)/2)
    points = {}
    landmark_names = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
    for i in range(n):
        x = int(landmarks[i])
        y = int(landmarks[i+n])
        points[landmark_names[i]] = (x,y)
    IPD = intra_point_distance(points)
    ED = euclidean(points['left_eye'], points['right_eye'])
    Ref_D = euclidean(points['left_eye'], points['left_mouth'])
    NED = ED/IPD
    if (NED<NED_threshold):
        return True
    else:
        return False

def check_if_all_landmarks_in_img(landmarks, bb):
    flag = True
    n = int(len(landmarks)/2)
    for i in range(n):
        x = int(landmarks[i])
        y = int(landmarks[i+n])
        if not bb[0] <= x <= bb[2]:
            flag = False
            return flag
        if not bb[1] <= y <= bb[3]:
            flag = False
            return flag
    return flag


def draw_line(img, P1, P2, tag, direction):
    p1 = (int(P1[0]), int(P1[1]))
    p2 = (int(P2[0]), int(P2[1]))
    cv2.line(img, p1, p2, (0, 255, 0), 1)
    offset_x = 0
    offset_y = 0
    if direction == 'vertical':
        offset_x = -10
    else:
        offset_y = -10
    P = (((p1[0]+p2[0])/2)+offset_x, ((p1[1]+p2[1])/2)+offset_y)
    mid_point = (int(P[0]), int(P[1]))
    cv2.putText(img, tag, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

def draw_point(img, P, tag):
    p = (int(P[0]), int(P[1]))
    cv2.circle(img, p, 1, (255, 0, 0), -1)
    cv2.putText(img, tag, (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

def get_mid_point(line):
    return ((line[0][0]+line[1][0])/2, (line[0][1]+line[1][1])/2)

def draw_bb(landmarks, bb, img, conf):
    #draw_bounding boxes and their confidences
    top_left = (bb[0], bb[1])
    top_right = (bb[2], bb[1])
    bottom_left = (bb[0], bb[3])
    bottom_right = (bb[2], bb[3])
    draw_line(img, top_left, top_right, str(round(conf, 3)), 'horizontal')
    draw_line(img, top_left, bottom_left, '', 'vertical')
    draw_line(img, top_right, bottom_right, '', 'vertical')
    draw_line(img, bottom_left, bottom_right, str(round(conf, 3)), 'horizontal')
    if (len(landmarks)!=0):
        n = int(len(landmarks)/2)
        points = {}
        landmark_names = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
        for i in range(n):
            x = int(landmarks[i])
            y = int(landmarks[i+n])
            points[landmark_names[i]] = (x,y)
        for tag, point in points.items():
            draw_point(img, point, '')


def get_distance_to_the_boundaries(landmarks, bb, img):
    top_left = (bb[0], bb[1])
    top_right = (bb[2], bb[1])
    bottom_left = (bb[0], bb[3])
    bottom_right = (bb[2], bb[3])
    upper_boundary = [top_left, top_right]
    left_boundary = [top_left, bottom_left]
    right_boundary = [top_right, bottom_right]
    bottom_boundary = [bottom_left, bottom_right]

    n = int(len(landmarks)/2)
    points = {}
    landmark_names = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
    for i in range(n):
        x = int(landmarks[i])
        y = int(landmarks[i+n])
        points[landmark_names[i]] = (x,y)
    info = {}#0-> distance, 1-> line, 2-> direction
    info['left_eye_to_upper_b'] = [get_norm_distance(upper_boundary, points['left_eye']), [points['left_eye'], (points['left_eye'][0], top_left[1])], 'vertical']
    info['right_eye_to_upper_b'] = [get_norm_distance(upper_boundary, points['right_eye']), [points['right_eye'], (points['right_eye'][0], top_right[1])], 'vertical']
    info['left_eye_to_left_b'] = [get_norm_distance(left_boundary, points['left_eye']), [points['left_eye'], (top_left[0], points['left_eye'][1])], 'horizontal']
    info['right_eye_to_right_b'] = [get_norm_distance(right_boundary, points['right_eye']), [points['right_eye'], (top_right[0], points['right_eye'][1])], 'horizontal']
    info['left_mouth_to_bottom_b'] = [get_norm_distance(bottom_boundary, points['left_mouth']), [points['left_mouth'], (points['left_mouth'][0], bottom_left[1])], 'vertical']
    info['right_mouth_to_bottom_b'] = [get_norm_distance(bottom_boundary, points['right_mouth']), [points['right_mouth'], (points['right_mouth'][0], bottom_right[1])], 'vertical']
    info['left_mouth_to_left_b'] = [get_norm_distance(left_boundary, points['left_mouth']), [points['left_mouth'], (top_left[0], points['left_mouth'][1])], 'horizontal']
    info['right_mouth_to_right_b'] = [get_norm_distance(right_boundary, points['right_mouth']), [points['right_mouth'], (top_right[0], points['right_mouth'][1])], 'horizontal']
    if img.shape!=(720, 1280, 3):
        for k, v in info.items():
            info[k][0] = v[0]*2
    return info

def flag_bad_landmarks(info):
    distances = [item[0] for item in list(info.values())]
    return all(i >= 45 for i in distances)
def resize_with_aspect_ratio(image, desired_size):
    im = Image.fromarray(image)
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image
    # thumbnail is a in-place operation
    # im.thumbnail(new_size, Image.ANTIALIAS)
    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,(desired_size-new_size[1])//2))
    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(im, padding)
    return np.asarray(new_im)

def crop_eyes(img, bb, landmarks, eye_patch_th = 0.1, output_file='', new_height=None, new_width=None):
    n = int(len(landmarks)/2)
    points = {}
    landmark_names = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
    for i in range(n):
        x = int(landmarks[i])
        y = int(landmarks[i+n])
        points[landmark_names[i]] = (x,y)
    avg_eye_line_y = (float(points['left_eye'][1])+points['right_eye'][1])/2
    bb_height = bb[3]-bb[1]
    new_bb = bb
    new_bb[1] = int(avg_eye_line_y - eye_patch_th*avg_eye_line_y)
    new_bb[3] = int(avg_eye_line_y + eye_patch_th*avg_eye_line_y)
    eye_patch = img[bb[1]:bb[3],bb[0]:bb[2],:]
    misc.imsave(output_file, eye_patch)
    return eye_patch

def get_norm_distance(line,point):
    #get_perpendicular_distance_from_point_to_line
    return norm(np.cross(np.array(line[1])-np.array(line[0]), np.array(line[0])-np.array(point))/norm(np.array(line[1])-np.array(line[0])))

def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    parent_dir = '/'.join(output_dir.split('/')[:-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = facenet.get_dataset(args.input_dir)

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face_ovi.create_mtcnn(sess, None)

    minsize = 100 # minimum size of face
    threshold = args.three_step_threshold
    #threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    distance_info = {}
    parent_dir = '/'.join(args.output_dir.split('/')[:-1])
    distance_info_file = parent_dir + '/distance_info.json'
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    if args.random_order:
        random.shuffle(dataset)
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
            if args.random_order:
                random.shuffle(cls.image_paths)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename+'.png')
            #print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                if img.ndim<2:
                    create_dir(parent_dir+'/failed_only_two_channels')
                    img = cv2.imread(image_path)
                    #img.save(parent_dir+'/failed_only_two_channels/'+filename+'.png')
                    #misc.imsave(parent_dir+'/failed_only_two_channels/'+filename+'.png', img)
                    cv2.imwrite(parent_dir+'/failed_only_two_channels/'+filename+'.png', img)
                    print('Unable to align "%s"' % image_path)
                    #text_file.write('%s\n' % (output_filename))
                    continue
                elif img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:,:,0:3]

                bounding_boxes, landmark_points, final_info = align.detect_face_ovi.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                #print (landmark_points)
                #landmark_points = landmark_points.tolist()
                if not np.all(landmark_points):
                    #print ('Unable to align due to less number of landmarks: ', landmark_points, image_path)
                    continue
                #draw_landmarks(img, landmark_points)
                #if (extract_image_chips.extract_image_chips(img,np.transpose(points), args.image_size, args.image_size, 0.37)):
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
                            #det = det[index,:]
                        det = np.squeeze(det)
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0]-args.margin/2, 0)
                        bb[1] = np.maximum(det[1]-args.margin/2, 0)
                        bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                        bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                        if args.visualize:
                            try:
                                draw_bb(landmark_points[:, i], bb, img, conf)
                            except IndexError:
                                draw_bb(landmark_points[-len(landmark_points)%10:], bb, img, conf)
                        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                        #if (cropped.shape[0]*cropped.shape[1]*cropped.shape[2]>biggest_face_shape[0]*biggest_face_shape[1]*biggest_face_shape[2]) and (args.th>conf>0.70):
                        #    if args.save_rejected:
                        #        t_scaled = misc.imresize(cropped, (args.image_height, args.image_width), interp='bilinear')
                        #        misc.imsave('/home/ovuser/Desktop/rejected/'+str(round(conf, 3))+'_'+str(image_path.split('/')[-1]), t_scaled)
                        #elif (cropped.shape[0]*cropped.shape[1]*cropped.shape[2]>biggest_face_shape[0]*biggest_face_shape[1]*biggest_face_shape[2]) and (conf>=args.th):
                        if (conf>best_face_conf):
                            best_face_conf = conf
                            best_face_index = i
                            best_face_shape = cropped.shape
                            best_face_cropped = cropped
                            best_face_bb = bb
                    bfi = best_face_index
                    if args.view_mtcnn_scores:
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
                    if args.filter_sideways:
                        sideways_flag = flag_sideways(best_face_bb, landmarks, args.NED_threshold)
                        if sideways_flag:
                            bb = best_face_bb
                            best_face_cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                            if args.keep_aspect_ratio:
                                scaled = resize_with_aspect_ratio(best_face_cropped, args.image_height)
                            else:
                                scaled = misc.imresize(best_face_cropped, (args.image_height, args.image_width), interp='bilinear')
                            create_dir(parent_dir+'/_sideways_faces')
                            misc.imsave(parent_dir+'/_sideways_faces/'+filename+'.png', scaled)
                            continue
                    if not l_flag:
                        bb = best_face_bb
                        best_face_cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                        if args.keep_aspect_ratio:
                            scaled = resize_with_aspect_ratio(best_face_cropped, args.image_height)
                        else:
                            scaled = misc.imresize(best_face_cropped, (args.image_height, args.image_width), interp='bilinear')
                        create_dir(parent_dir+'/_failed_landmarks_out_of_bounding_box')
                        misc.imsave(parent_dir+'/_failed_landmarks_out_of_bounding_box/'+filename+'.png', scaled)
                        continue
                    info = get_distance_to_the_boundaries(landmarks, best_face_bb, img)
                    if not flag_bad_landmarks(info) and args.discard_border_landmark_faces:
                        bb = best_face_bb
                        best_face_cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                        if args.keep_aspect_ratio:
                            scaled = resize_with_aspect_ratio(best_face_cropped, args.image_height)
                        else:
                            scaled = misc.imresize(best_face_cropped, (args.image_height, args.image_width), interp='bilinear')
                        create_dir(parent_dir+'/_failed_close_to_border_landmarks')
                        misc.imsave(parent_dir+'/_failed_close_to_border_landmarks/'+filename+'.jpg', scaled)
                        continue
                    distance_info[image_path] = [value[0] for value in list(info.values())]
                    distance_info[image_path].extend(list(img.shape))
                    bb = best_face_bb
                    best_face_cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    if args.get_eye_patches:
                        eye_patch_folder = parent_dir+'/eye_patches'
                        create_dir(eye_patch_folder)
                        eye_patch_file = eye_patch_folder+'/'+filename+'.jpg'
                        eye_patch = crop_eyes(img, bb, landmarks, eye_patch_th = args.eye_patch_th, output_file = eye_patch_file)
                    if args.keep_aspect_ratio:
                        scaled = resize_with_aspect_ratio(best_face_cropped, args.image_height)
                    else:
                        scaled = misc.imresize(best_face_cropped, (args.image_height, args.image_width), interp='bilinear')
                    if best_face_conf>args.th:
                        nrof_successfully_aligned += 1
                        if (not args.visualize):
                            misc.imsave(output_filename, scaled)
                        else:
                            misc.imsave(output_filename, img)
                    else:
                        create_dir(parent_dir+'/_rejected_due_to_conf_'+str(args.th))
                        misc.imsave(parent_dir+'/_rejected_due_to_conf_'+str(args.th)+'/'+filename+'.jpg', scaled)
                else:
                    print('Unable to align "%s"' % image_path)
                    create_dir(parent_dir+'/_no_initial_detection')
                    misc.imsave(parent_dir+'/_no_initial_detection/'+filename+'.jpg', img)
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--discard_border_landmark_faces', type=int, help='flag whether to discard landmark faces or not', default=0)
    parser.add_argument('--get_eye_patches', type=int, help='flag whether to save eye patches', default=0)
    parser.add_argument('--eye_patch_th', type=float,
        help='The eye_patch width parameter', default=0.1)
    parser.add_argument('--image_width', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--keep_aspect_ratio', type=int,
        help='Flag whether to keep aspect ratio while resizing', default=0)
    parser.add_argument('--image_height', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--visualize', type = int, help='Visualize bbs', default=0)
    parser.add_argument('--save_rejected', type = int, help='flag whether to save the rejected face pics', default=0)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
    parser.add_argument('--random_order',
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.5)
    parser.add_argument('--th', type=float,
        help='MTCNN bbox conf threshold', default=0.98)
    parser.add_argument('--three_step_threshold', type=float, nargs='+',
        help='MTCNN bbox conf threshold', default=[0.6, 0.7, 0.98])
    parser.add_argument('--filter_sideways', type=int, help='option to turn on sideways filtering', default=0)
    parser.add_argument('--NED_threshold', type=float, help='threshold value for sideways filtering if turned on', default=0.3)
    parser.add_argument('--view_mtcnn_scores', type=int, help='option to turn on MTCNN three step score convention in naming of the aligned images', default=0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
