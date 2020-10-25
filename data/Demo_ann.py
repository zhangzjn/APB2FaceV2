import os.path
import random
from PIL import Image
import torch
from torch.utils.data import dataset
import torchvision.transforms as transforms
import random


class Dataset_(dataset.Dataset):
    def __init__(self, opt):
        self.data_root = opt.data_root
        self.img_size = opt.img_size

        self.ref_idt = opt.ref_idt
        self.aud_idt = opt.aud_idt
        self.own_audio = opt.own_audio
        self.non_ref = opt.non_ref

        idts = os.listdir(self.data_root)
        self.data_all = {}
        exclusion = ['jaime', '256_feature']
        for ex in exclusion:
            if ex in idts:
                idts.remove(ex)
        idts.sort()
        for idt in idts:
            data_all = []
            idt_path = '{}/{}'.format(self.data_root, idt)
            for mode in ['train', 'test']:
                idt_pack = '{}/{}_{}.t7'.format(idt_path, self.img_size, mode)
                idt_files = torch.load(idt_pack)
                img_paths = idt_files['img_paths']
                aud_feats = idt_files['audio_features']
                lands = idt_files['lands']
                poses = idt_files['poses']
                eyes = idt_files['eyes']
                for i in range(len(img_paths)):
                    img_abs_path = '{}/{}'.format(idt_path, img_paths[i][0])  # [image, landmark]
                    aud_feature = aud_feats[i]
                    land = lands[i]
                    pose = poses[i]
                    eye = eyes[i]
                    data_all.append([img_abs_path, aud_feature, land, pose, eye])
            data_all.sort(key=lambda x: int(x[0].split('/')[-1].split('.')[0]))
            # random.shuffle(data_all)
            self.data_all[idt] = data_all

        self.transforms_image = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        idt2ref = {'man1': '323.jpg', 'man2': '155.jpg', 'man3': '80.jpg',
                   'woman1': '261.jpg', 'woman2': '1932.jpg', 'woman3': '86.jpg'}

        self.data2 = self.find_ref(idt2ref[self.ref_idt], self.data_all[self.ref_idt])

    def find_ref(self, img_name, data_all):
        for data_ in data_all:
            if data_[0].split('/')[-1] == img_name:
                return data_
        return None


    def __len__(self):
        return len(self.data_all[self.aud_idt])


    def __getitem__(self, index):
        data1 = self.data_all[self.aud_idt][index]
        data2 = self.data2
        img_path1, aud_feat1, land1, pose1, eye1 = data1
        img_path2, aud_feat2, land2, pose2, eye2 = data2
        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')
        img1 = self.transforms_image(img1)
        img2 = self.transforms_image(img2)
        aud_feat1 = torch.tensor(aud_feat1).unsqueeze(dim=0)
        pose1 = torch.tensor(pose1)
        eye1 = torch.tensor(eye1)
        return [aud_feat1, pose1, eye1, img1, img2]
