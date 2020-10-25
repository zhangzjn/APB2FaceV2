import torch
import torch.nn as nn


class AudioNet(nn.Module):

    def __init__(self, num_landmark=212):
        super(AudioNet, self).__init__()
        self.num_landmark = num_landmark
        # audio
        self.audio1 = nn.Sequential(
            nn.Conv2d(1, 72, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
        )
        self.audio2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(4, 1), stride=(4, 1)), nn.ReLU()
        )
        self.trans_audio = nn.Sequential(nn.Linear(256 * 2, 256))
        # pose
        self.trans_pose = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        # eye
        self.trans_eye = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        # cat
        self.trans_cat1 = nn.Sequential(
            nn.Linear(256 + 64 * 2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.trans_cat2 = nn.Sequential(
            nn.Linear(256, self.num_landmark)
        )

    def norm_1d(self, x):
        eps = 1e-5
        gamma = torch.ones(x.shape[0]).to(x.device)
        beta = torch.zeros(x.shape[0]).to(x.device)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_var = torch.mean((x - x_mean) ** 2, dim=1, keepdim=True)
        x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
        return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, audio, pose, eye):
        x_a = self.audio1(audio)
        x_a = self.audio2(x_a)
        x_a = x_a.view(-1, self.num_flat_features(x_a))
        x_a = self.trans_audio(x_a)
        x_p = self.trans_pose(pose)
        x_e = self.trans_eye(eye)
        x_cat = torch.cat([x_a, x_p, x_e], dim=1)
        x_cat = self.trans_cat1(x_cat)
        latent = self.norm_1d(x_cat)
        out = self.trans_cat2(latent)
        return latent, out


if __name__ == "__main__":
    audio = torch.randn(2, 1, 64, 60)
    pose = torch.randn(2, 3)
    eye = torch.randn(2, 2)
    net = AudioNet()
    out0, out1 = net(audio, pose, eye)
    print(out0.shape, out1.shape)