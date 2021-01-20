import python_speech_features as psf
import numpy as np
import librosa
import scipy.io.wavfile as wav
import os
import random
from tqdm import tqdm
import glob
import cv2
from data_utils import crop_and_generate_landmark_img, crop_and_generate_landmark_img_two_size
import torch


def split_video(video_file, label_file, out_path):
    labels = [[float(l_) for l_ in l.strip().split()]for l in open(label_file, 'r').readlines()]
    out_path = os.path.join(out_path, 'image')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    cap = cv2.VideoCapture(video_file)
    frame_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for label in labels:
        l, r = label
        cap.set(cv2.CAP_PROP_POS_FRAMES, l)
        cnt = int(l)
        while cnt <= r:
            print('\r{} {}/{}'.format(video_file, cnt, frame_len), end='')
            ret, img = cap.read()
            # img = np.rot90(img, 1)
            if img is None:
                print('frame {} error'.format(cnt))
                continue
            cv2.imwrite('{}/{}.jpg'.format(out_path, cnt), img)
            cnt += 1
        print()


def feature_extraction_AnnVI(video_file, audio_file, label_file, out_path, img_size=256):
    # audio setting
    n_fft = 2048  # 44100/30 1470
    hop_length = 512  # 44100/60 735
    n_mfcc = 20
    sr = 44100
    win_size = 64
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # load / solve data
    sig, rate = librosa.load(audio_file, sr=sr, duration=None)
    time_duration = len(sig) / rate
    # print('time of duration : {}'.format(time_duration))l
    f_mfcc = librosa.feature.mfcc(sig, rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    f_mfcc_delta = psf.base.delta(f_mfcc, 2)
    f_mfcc_delta2 = psf.base.delta(f_mfcc_delta, 2)
    f_mfcc_all = np.concatenate((f_mfcc, f_mfcc_delta, f_mfcc_delta2), axis=0)

    # landmarks
    land_file = '{}/{}_landmark_crop.txt'.format(out_path, img_size)
    landmarks = open(land_file).readlines()
    landmarks.sort(key=lambda x: int(x.strip().split()[0].split('.')[0]))
    lands_all = dict()
    poses_all = dict()
    eyes_all = dict()
    for landmark in landmarks:
        l = landmark.strip().split()
        name = l[0]
        w_ori, h_ori = [int(_) for _ in l[1].split('-')]
        land = []
        for l_ in l[2:108]:
            w, h = [float(_) for _ in l_.split('-')]
            land.extend([w, h])
        pose = []
        for l_ in l[108:111]:
            pose.append(float(l_))
        eye = []
        for l_ in l[111:113]:
            eye.append(float(l_))
        lands_all[name.split('.')[0]] = land
        poses_all[name.split('.')[0]] = pose
        eyes_all[name.split('.')[0]] = eye
        # land = np.array(land).astype(np.float64)


    labels = [[float(l_) for l_ in l.strip().split()] for l in open(label_file, 'r').readlines()]
    img_paths = []
    audio_features = []
    lands = []
    poses = []
    eyes = []
    for label in labels:
        l, r = label
        print('\r{} - {}'.format(l, r))
        cnt = int(l)
        while cnt <= r:
            img_path = '{}_image_crop/{}.jpg'.format(img_size, cnt)
            land_path = '{}_landmark_crop/{}.jpg'.format(img_size, cnt)
            if not os.path.exists(os.path.join(out_path, img_path)):
                print('{} not exists'.format(img_path))
                cnt += 1
                continue
            img_paths.append([img_path, land_path])
            c_count = int(cnt / fps * rate / hop_length)
            audio_features.append(f_mfcc_all[:, c_count - win_size // 2: c_count + win_size // 2].transpose(1, 0))
            lands.append(lands_all['{}'.format(cnt)])
            poses.append(poses_all['{}'.format(cnt)])
            eyes.append(eyes_all['{}'.format(cnt)])
            cnt += 1e
    ratio = 0.9
    length = len(img_paths)
    length_train = int(length * ratio)
    index = list(range(length))
    random.shuffle(index)
    # train dataset
    img_paths_train = [img_paths[i] for i in index[:length_train]]
    audio_features_train = [audio_features[i] for i in index[:length_train]]
    lands_train = [lands[i] for i in index[:length_train]]
    poses_train = [poses[i] for i in index[:length_train]]
    eyes_train = [eyes[i] for i in index[:length_train]]
    # test dataset
    img_paths_test = [img_paths[i] for i in index[length_train:]]
    audio_features_test = [audio_features[i] for i in index[length_train:]]
    lands_test = [lands[i] for i in index[length_train:]]
    poses_test = [poses[i] for i in index[length_train:]]
    eyes_test = [eyes[i] for i in index[length_train:]]

    save_data_train = {'img_paths': img_paths_train, 'audio_features': audio_features_train, 'lands': lands_train,
                       'poses': poses_train, 'eyes': eyes_train}
    torch.save(save_data_train, os.path.join(out_path, '{}_train.t7'.format(img_size)))
    save_data_test = {'img_paths': img_paths_test, 'audio_features': audio_features_test, 'lands': lands_test,
                      'poses': poses_test, 'eyes': eyes_test}
    torch.save(save_data_test, os.path.join(out_path, '{}_test.t7'.format(img_size)))


def main_AnnVI(dataroot):
    feature_path = '{}/feature'.format(dataroot)
    if not os.path.isdir(feature_path): os.mkdir(feature_path)

    video_files = glob.glob('{}/video/*.mp4'.format(dataroot))

    steps = [1, 2, 3]
    if 1 in steps:
        for video_file in video_files:
            label_file = video_file.replace('video', 'label').replace('mp4', 'txt')
            if not os.path.exists(label_file):
                continue
            out_path = os.path.join(feature_path, os.path.basename(video_file).split('.')[0])
            if not os.path.isdir(out_path): os.mkdir(out_path)
            split_video(video_file, label_file, out_path)
            print('{} solved'.format(video_file))
    if 2 in steps:
        for video_file in video_files:
            name_ = os.path.basename(video_file).split('.')[0]
            out_path = os.path.join(feature_path, name_)
            if not os.path.isdir(out_path): os.mkdir(out_path)
            landmark_path = out_path.replace('feature', 'landmark').replace(name_, '{}.txt'.format(name_))
            if not os.path.exists(landmark_path):
                print('skip {}'.format(landmark_path))
                continue
            crop_and_generate_landmark_img_two_size(out_path, landmark_path)
            print('{} solved'.format(video_file))
    if 3 in steps:
        for video_file in video_files:
            name_ = os.path.basename(video_file).split('.')[0]
            out_path = os.path.join(feature_path, name_)
            landmark_path = out_path.replace('feature', 'landmark').replace(name_, '{}.txt'.format(name_))
            if not os.path.exists(landmark_path):
                print('skip {}'.format(landmark_path))
                continue
            audio_file = video_file.replace('video', 'audio').replace('mp4', 'm4a')
            label_file = video_file.replace('video', 'label').replace('mp4', 'txt')
            if not os.path.exists(label_file):
                continue
            if not os.path.isdir(out_path): os.mkdir(out_path)
            feature_extraction_AnnVI(video_file, audio_file, label_file, out_path, img_size=256)
            feature_extraction_AnnVI(video_file, audio_file, label_file, out_path, img_size=512)
            print('{} solved'.format(video_file))


def count_frames(dataroot):
    files = glob.glob('{}/label/*.txt'.format(dataroot))
    cnts = dict()
    for file in files:
        cnt = 0
        labels = [[int(l_) for l_ in l.strip().split()] for l in open(file, 'r').readlines()]
        for label in labels:
            cnt += (label[1] - label[0])
        cnts[file] = cnt
    cnt_all = 0
    for key, val in cnts.items():
        cnt_all += val
        print('{:<10}: \t{}'.format(os.path.basename(key), val))
    print('{:<10}: \t{}'.format('Total', cnt_all))


if __name__ == '__main__':
    dataroot = '/media/zhangzjn/datasets/AnnTest/'
    main_AnnVI(dataroot)
    # count_frames(dataroot)
