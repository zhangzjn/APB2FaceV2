import random
import collections
from torch.utils.data import sampler


class VCSampler(sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_img):
        super(VCSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_id = batch_id
        self.batch_img = batch_img

        self.idts = data_source.idts
        self.data_all = data_source.data_all
        self.idt2idxes = data_source.idt2idxes
        self.idt_valid = len(self.idts) // self.batch_id * self.batch_id

    def __iter__(self):
        random.shuffle(self.idts)
        imgs = []
        for i in range(self.idt_valid):
            imgs.extend(self._sample(self.idt2idxes[self.idts[i]], self.batch_img))
        return iter(imgs)

    def __len__(self):
        return self.idt_valid * self.batch_img

    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        return random.sample(population, k)