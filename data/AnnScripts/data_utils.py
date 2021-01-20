import numpy as np
from skimage import io, transform
import cv2
from PIL import Image
import copy
import os, sys
import random


def tuple_shape(shape):
    r_data = []
    for p in shape:
        r_data.append([p.x, p.y])
    return r_data

def drawCircle(img, shape, radius=1, color=(255, 255, 255), thickness=1):
    for i, p in enumerate(shape):
        img = cv2.circle(img, (int(p[0]), int(p[1])), radius, color, thickness)
        # img = cv2.putText(img, '{}'.format(i), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    return img

def drawPoint(img, shape):
    for i, p in enumerate(shape):
        # img[int(p[0]), int(p[1])] = (255, 255, 255)
        img[int(p[1]), int(p[0])] = (255, 255, 255)
    return img

def fillPoly(img, shape, color=(255, 255, 255)):
    shape_poly = []
    for i in range(0, 33):
        p = shape[i]
        shape_poly.append([int(p[0]), int(p[1])])
    for i in range(41, 33, -1):
        p = shape[i]
        shape_poly.append([int(p[0]), int(p[1])])
    img = cv2.fillPoly(img, np.array([shape_poly], dtype=np.int32), color)
    img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=10)
    return img

# for *.txt output
def str_landmark(land_in, name, img_size, pose, eye):
    out_landmark = ''
    out_landmark += '{}'.format(name)
    out_landmark += ' {}-{}'.format(img_size[0], img_size[1])
    for p in land_in:
        out_landmark += ' {}-{}'.format(p[0], p[1])
    for p in pose:
        out_landmark += ' {}'.format(p)
    for p in eye:
        out_landmark += ' {}'.format(p)
    out_landmark += '\n'
    return out_landmark

def str_landmark_only(land_in, name, img_size):
    out_landmark = ''
    for p in land_in:
        out_landmark += '{}-{} '.format(p[0], p[1])
    out_landmark += '\n'
    return out_landmark

def str_landmark_only1(land_in, name, img_size):
    out_landmark = ''
    for p in land_in:
        out_landmark += '{} {} '.format(p[0], p[1])
    out_landmark += '\n'
    return out_landmark


def crop_and_generate_landmark_img(base):

    def xy(shape):
        x_min, x_max, y_min, y_max = shape[0][0], shape[0][0], shape[0][1], shape[0][1]
        for p in shape:
            x_min = min([x_min, p[0]])
            x_max = max([x_max, p[0]])
            y_min = min([y_min, p[1]])
            y_max = max([y_max, p[1]])
        return (x_min + x_max) / 2, (y_min + y_max) / 2, max([(x_max - x_min) / 2, (y_max - y_min) / 2])

    def generate():
        labs_ = open(land_txt, 'r').readlines()
        f = open(land_norm_txt, 'w')
        for _, l in enumerate(labs_):
            print('\r{} {}/{}'.format(land_txt, _ + 1, len(labs_)), end='')
            l = l.strip().split()
            name = l[0]
            w_ori, h_ori = [int(_) for _ in l[1].split('-')]
            shape = []
            for l_ in l[2:108]:
                w, h = [float(_) for _ in l_.split('-')]
                shape.append([w, h])
            pose = []
            for l_ in l[108:111]:
                pose.append(float(l_))
            eye = []
            for l_ in l[111:113]:
                eye.append(float(l_))
            img_size = 512
            p = 1.4
            x_c, y_c, r = xy(shape)
            ll = r * p
            zoom_p = (2 * ll) / img_size
            x_o, y_o = x_c - ll, y_c - ll
            img_path = os.path.join(image_path, name)
            if not os.path.exists(img_path):
                print('img {} not exists'.format(img_path))
                continue
            img = cv2.imread(img_path)

            img_roi = img[int(y_o):int(y_o + 2*ll), int(x_o):int(x_o + 2*ll), :].copy()

            for i in range(len(shape)):
                s = shape[i]
                shape[i] = [(s[0] - x_o) / zoom_p, (s[1] - y_o) / zoom_p]
            if img_roi.size == 0:
                continue
            img_crop = cv2.resize(img_roi, (img_size, img_size))
            lab_template = np.zeros((img_size, img_size, 3))
            img_show = drawCircle(img_crop.copy(), shape, radius=2, color=(0, 0, 255), thickness=8)
            land_show = drawCircle(lab_template.copy(), shape, radius=1, color=(255, 255, 255), thickness=8)
            land_crop_thick = drawCircle(lab_template.copy(), shape, radius=2, color=(255, 255, 255), thickness=8)
            land_crop_thin = drawPoint(lab_template.copy(), shape)
            land_fill = fillPoly(lab_template.copy(), shape, color=(255, 255, 255))

            f.write(str_landmark(shape, name, (img_size, img_size), pose, eye))

            cv2.imwrite('{}/{}'.format(image_path_crop, name), img_crop)
            cv2.imwrite('{}/{}'.format(image_path_show, name), img_show)
            cv2.imwrite('{}/{}'.format(land_path_show, name), land_show)
            cv2.imwrite('{}/{}'.format(land_path_show_thick, name), land_crop_thick)
            cv2.imwrite('{}/{}'.format(land_path_show_thin, name), land_crop_thin)
            cv2.imwrite('{}/{}'.format(land_path_fill, name), land_fill)
        f.close()

    image_path = os.path.join(base, 'image')
    image_path_crop = os.path.join(base, 'image_crop')
    image_path_show = os.path.join(base, 'image_show')
    land_path_show = os.path.join(base, 'landmark_crop')
    land_path_show_thick = os.path.join(base, 'landmark_crop_thick')
    land_path_show_thin = os.path.join(base, 'landmark_crop_thin')
    land_path_fill = os.path.join(base, 'landmark_fill')
    land_txt = os.path.join(base, 'landmark.txt')
    land_norm_txt = os.path.join(base, 'landmark_crop.txt')
    if not os.path.exists(image_path_crop):
        os.mkdir(image_path_crop)
        os.mkdir(image_path_show)
        os.mkdir(land_path_show)
        os.mkdir(land_path_fill)
        os.mkdir(land_path_show_thick)
        os.mkdir(land_path_show_thin)
    generate()


def crop_and_generate_landmark_img_two_size(base, landmark_path):

    def xy(shape):
        x_min, x_max, y_min, y_max = shape[0][0], shape[0][0], shape[0][1], shape[0][1]
        for p in shape:
            x_min = min([x_min, p[0]])
            x_max = max([x_max, p[0]])
            y_min = min([y_min, p[1]])
            y_max = max([y_max, p[1]])
        return (x_min + x_max) / 2, (y_min + y_max) / 2, max([(x_max - x_min) / 2, (y_max - y_min) / 2])
        # return x_min, y_min, max([(x_max - x_min) / 2, (y_max - y_min) / 2])

    def generate(img_size=512, thin_only=False):
        image_path = os.path.join(base, 'image')
        image_crop_path = '{}/{}_image_crop'.format(base, img_size)
        image_show_path = '{}/{}_image_show'.format(base, img_size)
        landmark_crop_path = '{}/{}_landmark_crop'.format(base, img_size)
        landmark_crop_thick_path = '{}/{}_landmark_crop_thick'.format(base, img_size)
        landmark_crop_thin_path = '{}/{}_landmark_crop_thin'.format(base, img_size)
        landmark_fill_path = '{}/{}_landmark_fill'.format(base, img_size)
        # landmark_txt = '{}/landmark.txt'.format(base)
        landmark_txt = landmark_path
        landmark_crop_txt = '{}/{}_landmark_crop.txt'.format(base, img_size)
        if not thin_only:
            if not os.path.exists(image_crop_path):
                os.mkdir(image_crop_path)
                os.mkdir(image_show_path)
                os.mkdir(landmark_crop_path)
                os.mkdir(landmark_crop_thick_path)
                os.mkdir(landmark_fill_path)
                os.mkdir(landmark_crop_thin_path)
        else:
            os.mkdir(image_crop_path)
            os.mkdir(landmark_crop_thin_path)
        labs_ = open(landmark_txt, 'r').readlines()
        f = open(landmark_crop_txt, 'w')
        for _, l in enumerate(labs_):
            print('\r{} {}/{}'.format(landmark_txt, _ + 1, len(labs_)), end='')
            l = l.strip().split()
            name = l[0]
            w_ori, h_ori = [int(_) for _ in l[1].split('-')]
            shape = []
            for l_ in l[2:108]:
                w, h = [float(_) for _ in l_.split('-')]
                shape.append([w, h])
            pose = []
            for l_ in l[108:111]:
                pose.append(float(l_))
            eye = []
            for l_ in l[111:113]:
                eye.append(float(l_))
            # img_size = 512
            p = 1.4
            x_c, y_c, r = xy(shape)
            ll = r * p
            zoom_p = (2 * ll) / img_size
            x_o, y_o = x_c - ll, y_c - ll
            img_path = os.path.join(image_path, name)
            if not os.path.exists(img_path):
                print('img {} not exists'.format(img_path))
                continue
            img = cv2.imread(img_path)

            img_roi = img[int(y_o):int(y_o + 2*ll), int(x_o):int(x_o + 2*ll), :].copy()

            for i in range(len(shape)):
                s = shape[i]
                shape[i] = [(s[0] - x_o) / zoom_p, (s[1] - y_o) / zoom_p]
            if img_roi.size == 0:
                continue
            img_crop = cv2.resize(img_roi, (img_size, img_size))
            lab_template = np.zeros((img_size, img_size, 3))
            img_show = drawCircle(img_crop.copy(), shape, radius=2, color=(0, 0, 255), thickness=8)
            land_crop = drawCircle(lab_template.copy(), shape, radius=1, color=(255, 255, 255), thickness=8)
            landmark_crop_thick = drawCircle(lab_template.copy(), shape, radius=2, color=(255, 255, 255), thickness=8)
            landmark_crop_thin = drawPoint(lab_template.copy(), shape)
            landmark_fill = fillPoly(lab_template.copy(), shape, color=(255, 255, 255))

            f.write(str_landmark(shape, name, (img_size, img_size), pose, eye))
            if not thin_only:
                cv2.imwrite('{}/{}'.format(image_crop_path, name), img_crop)
                cv2.imwrite('{}/{}'.format(image_show_path, name), img_show)
                cv2.imwrite('{}/{}'.format(landmark_crop_path, name), land_crop)
                cv2.imwrite('{}/{}'.format(landmark_crop_thick_path, name), landmark_crop_thick)
                cv2.imwrite('{}/{}'.format(landmark_crop_thin_path, name), landmark_crop_thin)
                cv2.imwrite('{}/{}'.format(landmark_fill_path, name), landmark_fill)
            else:
                cv2.imwrite('{}/{}'.format(image_crop_path, name), img_crop)
                cv2.imwrite('{}/{}'.format(landmark_crop_thin_path, name), landmark_crop_thin)
        f.close()

    generate(img_size=512)
    generate(img_size=256, thin_only=True)
