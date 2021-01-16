import os.path
import random
from PIL import Image
import torch
from torch.utils.data import dataset
import torchvision.transforms as transforms


class Dataset_(dataset.Dataset):
    def __init__(self, opt):
        self.data_root = opt.data_root
        self.mode = opt.mode
        self.idt_test = opt.idt_test
        self.img_size = opt.img_size
        self.own_audio = opt.own_audio
        self.non_ref = opt.non_ref

        self.idts = os.listdir(self.data_root)
        self.data_all, self.idt2idxes = [], {}
        exclusion = ['jaime', '256_feature']
        for ex in exclusion:
            if ex in self.idts:
                self.idts.remove(ex)
        self.idts.sort()
        cnt = 0
        for idt in self.idts:
            idxes = list()
            idt_path = '{}/{}'.format(self.data_root, idt)
            idt_pack = '{}/{}_{}.t7'.format(idt_path, self.img_size, self.mode)
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
                self.data_all.append([img_abs_path, aud_feature, land, pose, eye])
                idxes.append(cnt)
                cnt += 1
            self.idt2idxes[idt] = idxes

        self.transforms_image = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if self.mode == 'train':
            self.shuffle()
        else:
            if self.idt_test == 'random':
                self.idt_test = random.sample(self.idts, 1)[0]
            index_ = random.sample(self.idt2idxes[self.idt_test], 1)[0]
            self.data2 = self.data_all[index_]

    def shuffle(self):
        idxes = list(range(len(self.data_all)))
        random.shuffle(idxes)
        self.data_all = [self.data_all[idxes[i]] for i in range(len(idxes))]
        for idt, idt2idxes in self.idt2idxes.items():
            idt2idxes = [idxes.index(idx) for idx in idt2idxes]
            self.idt2idxes[idt] = idt2idxes

    def sort(self):
        if 'man' in ','.join(self.idts):  # AnnVI
            self.data_all.sort(key=lambda x: int(x[0].split('/')[-1].split('.')[0]))
        else:  # AnnXXX
            self.data_all.sort(key=lambda x: (int(x[0].split('/')[-1].split('.')[0].split('-')[0]),
                                              int(x[0].split('/')[-1].split('.')[0].split('-')[1])))

    def get_idt(self, img_path):
        img_idt = img_path.strip().split('/')[-3]
        return img_idt

    def __len__(self):
        if self.mode == 'train':
            return len(self.data_all)
        else:
            return len(self.idt2idxes[self.idt_test])

    def __getitem__(self, index):
        if self.mode == 'train':
            data1 = self.data_all[index]
            index_data2 = random.sample(self.idt2idxes[self.get_idt(data1[0])], 1)[0]
            data2 = self.data_all[index_data2]
            img_path1, aud_feat1, land1, pose1, eye1 = data1
            img_path2, aud_feat2, land2, pose2, eye2 = data2
            img1 = Image.open(img_path1).convert('RGB')
            img2 = Image.open(img_path2).convert('RGB')
            img1 = self.transforms_image(img1)
            img2 = self.transforms_image(img2)
            aud_feat1 = torch.tensor(aud_feat1).unsqueeze(dim=0)
            pose1 = torch.tensor(pose1)
            eye1 = torch.tensor(eye1)
        else:
            if self.own_audio:
                index_data1 = self.idt2idxes[self.idt_test][index]
                data1 = self.data_all[index_data1]
                img_path1, aud_feat1, land1, pose1, eye1 = data1
                img_path2, aud_feat2, land2, pose2, eye2 = self.data2
                img1 = Image.open(img_path1).convert('RGB')
                img2 = Image.open(img_path2).convert('RGB')
                img1 = self.transforms_image(img1)
                img2 = self.transforms_image(img2)
                aud_feat1 = torch.tensor(aud_feat1).unsqueeze(dim=0)
                pose1 = torch.tensor(pose1)
                eye1 = torch.tensor(eye1)
            else:
                idt1 = random.sample(self.idts, 1)[0]
                index_data1 = random.sample(self.idt2idxes[idt1], 1)[0]
                data1 = self.data_all[index_data1]
                img_path1, aud_feat1, land1, pose1, eye1 = data1
                img_path2, aud_feat2, land2, pose2, eye2 = self.data2
                img1 = Image.open(img_path1).convert('RGB')
                img2 = Image.open(img_path2).convert('RGB')
                img1 = self.transforms_image(img1)
                img2 = self.transforms_image(img2)
                aud_feat1 = torch.tensor(aud_feat1).unsqueeze(dim=0)
                pose1 = torch.tensor(pose1)
                eye1 = torch.tensor(eye1)
        return [aud_feat1, pose1, eye1, img1, img2]


if __name__ == '__main__':
    from torch.utils.data import dataloader
    from util.options import get_opt
    from sampler import VCSampler
    opt = get_opt()
    opt.data = 'AnnVI'
    opt.data_root = '/media/datasets/zhangzjn/AnnVI/feature'
    # opt.data = 'AnnXXX'
    # opt.data_root = '/media/datasets/zhangzjn/AnnXXX/feature'
    opt.img_size = 256
    opt.mode = 'train'
    opt.non_ref = True
    opt.own_audio = True
    # opt.triplet = True
    dataset_loader = Dataset_(opt)
    dataset_loader = dataloader.DataLoader(dataset_loader,
                                           # sampler=VCSampler(dataset_loader, opt.batch_id, opt.batch_img),
                                           batch_size=opt.batch_size, num_workers=2)
    dataset_size = len(dataset_loader)
    print(dataset_size)
    for i, data in enumerate(dataset_loader):
        print(i)
        break
