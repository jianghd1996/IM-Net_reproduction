import os
import random
import mcubes as libmcubes
import trimesh
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import h5py
import json
from tqdm import tqdm
from tensorboardX import SummaryWriter

class IM_Net(nn.Module):
    def __init__(self, z_dim=128):
        super(IM_Net, self).__init__()

        self.get_encoder(n_layers=5, z_dim=z_dim)
        self.get_decoder(n_layers=6, z_dim = z_dim)

    def get_encoder(self, n_layers, z_dim):
        model = []

        in_channels = 1
        out_channels = 32

        for i in range(n_layers - 1):
            model.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=(4,4,4), stride=(2,2,2), padding=1, bias=False))
            model.append(nn.LeakyReLU())
            model.append(nn.BatchNorm3d(num_features=out_channels))

            in_channels = out_channels
            out_channels *= 2

        model.append(nn.Conv3d(in_channels, z_dim, kernel_size=(4, 4, 4), stride=(1, 1, 1), padding=0))
        model.append(nn.Sigmoid())

        self.Encoder = nn.Sequential(*model)

    def get_decoder(self, n_layers, z_dim):
        feature_dim = z_dim + 3
        in_channels = feature_dim
        out_channels = 2048

        model = []
        for i in range(n_layers - 1):
            model.append([nn.Linear(in_channels, out_channels), nn.Dropout(), nn.LeakyReLU()])

            if i < 4:
                in_channels = out_channels + feature_dim
            else:
                in_channels = out_channels
            out_channels = out_channels // 2

        model.append([nn.Linear(in_channels, 1), nn.Sigmoid()])

        self.layer1 = nn.Sequential(*model[0])
        self.layer2 = nn.Sequential(*model[1])
        self.layer3 = nn.Sequential(*model[2])
        self.layer4 = nn.Sequential(*model[3])
        self.layer5 = nn.Sequential(*model[4])
        self.layer6 = nn.Sequential(*model[5])

    def forward(self, data, points):
        z = self.Encoder(data.unsqueeze(1)).view(data.shape[0], -1)
        point_batch_size = points.shape[1]
        z = z.unsqueeze(1).repeat((1, point_batch_size, 1)).view(data.shape[0]*point_batch_size, -1)
        points = points.view(-1, 3)

        feature_z = torch.cat([points, z], dim=1)
        out = self.layer1(feature_z)
        out = self.layer2(torch.cat([out, feature_z], dim=1))
        out = self.layer3(torch.cat([out, feature_z], dim=1))
        out = self.layer4(torch.cat([out, feature_z], dim=1))
        out = self.layer5(torch.cat([out, feature_z], dim=1))
        out = self.layer6(out)
        return out

class TrainClock(object):
    def __init__(self):
        self.epoch = 0
        self.step = 0

    def tick(self):
        self.step += 1

    def tock(self):
        self.epoch += 1

    def make_ckpt(self):
        return {
            "epoch": self.epoch,
            "step" 	: self.step
        }

    def load_ckpt(self, ckpt):
        self.epoch = ckpt['epoch']
        self.step = ckpt['step']

class ShapeNet(data.Dataset):
    def __init__(self, data_path, point_batch_size, resolution, category, split_list):
        self.data = []
        self.pbz = point_batch_size
        self.data_path = os.path.join("processed_data", category)
        name_list = json.load(open(split_list, "r"))
        for i in range(len(name_list)):
            file = os.path.join(self.data_path, name_list[i]['anno_id']+".npy")
            if not os.path.exists(file):
                continue
            self.data.append(name_list[i]['anno_id'])
        self.resolution = resolution


    def __getitem__(self, index):
        data = np.load(os.path.join(self.data_path, self.data[index]+".npy"), allow_pickle=True)
        data = data[()]
        voxel, points = data[self.resolution]
        voxel = data[self.resolution][0]

        length = 64 // self.resolution
        voxel = np.repeat(voxel, length, axis=0)
        voxel = np.repeat(voxel, length, axis=1)
        voxel = np.repeat(voxel, length, axis=2)

        N = len(points)
        for i in range(N):
            delt = np.random.randn(3) * length - length / 2
            n_point = points[i] + delt
            if np.min(n_point) >= 0 and np.max(n_point) < 64:
                points.append(n_point)

        for i in range(max(0, (self.pbz - len(points)) + 10)):
            points.append(np.random.rand(3) * 64)

        random.shuffle(points)
        points = points[:self.pbz]
        label = []
        for i in range(self.pbz):
            x, y, z = int(points[i][0]), int(points[i][1]), int(points[i][2])
            label.append(voxel[x][y][z])
        return voxel, np.array(points, dtype="float32") / 64, np.array(label)

    def __len__(self):
        return len(self.data)

class Agent(object):
    def __init__(self, data_path, category):
        self.epoch = 300
        self.data_path = data_path
        self.net       = IM_Net().cuda()
        self.policies = self.net.parameters()
        self.clock      = TrainClock()

        self.optimizer = torch.optim.Adam(self.policies, lr=1e-3, betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.criterion = nn.MSELoss().cuda()

        self.log_dir = 'event'

        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

        self.model_dir = 'weight'
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.category = category

    def generate_dataset(self):
        epoch = [0, 20, 60, 100]
        res = [8, 16, 32, 64]
        point_batch_size = [256, 512, 4096, 8192]

        resolution = 0
        pbz = 0
        flag = 0
        for i in range(4):
            if self.clock.epoch >= epoch[i]:
                resolution = res[i]
                pbz = point_batch_size[i]
            if self.clock.epoch == epoch[i]:
                flag = 1

        if flag == 0:
            return

        self.train_loader = torch.utils.data.DataLoader(
            ShapeNet(os.path.join(self.data_path, self.category), point_batch_size=pbz, resolution=resolution, category=self.category,
                     split_list=os.path.join(self.data_path, "train_val_test_split", self.category + ".train.json")),
            batch_size=16, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            ShapeNet(os.path.join(self.data_path, self.category), point_batch_size=pbz, resolution=resolution, category=self.category,
                     split_list=os.path.join(self.data_path, "train_val_test_split", self.category + ".val.json")),
            batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=True
        )

        self.val_loader = cycle(self.val_loader)

    def train(self):
        val_frequency = 20
        for e in range(self.clock.epoch, self.epoch):
            print(e)
            self.generate_dataset()
            pbar = tqdm(self.train_loader)
            for b, data in enumerate(pbar):
                loss = self.train_func(data)
                self.update_network(loss)

                if self.clock.step % val_frequency == 0:
                    data = next(self.val_loader)
                    loss = self.val_func(data)

                self.clock.tick()

            if e % 2 == 0:
                self.save_ckpt()

            self.scheduler.step(self.clock.epoch)
            self.clock.tock()
            self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)

    def test(self):
        self.load_ckpt()
        self.test_loader = torch.utils.data.DataLoader(
            ShapeNet(os.path.join(self.data_path, self.category), point_batch_size=4096, resolution=64, category=self.category,
                     split_list=os.path.join(self.data_path, "train_val_test_split", self.category + ".test.json")),
            batch_size=4, shuffle=False, num_workers=8, pin_memory=True, drop_last=True
        )
        pbar = tqdm(self.test_loader)

        tot_loss = 0
        tot_cnt = 0
        for b, data in enumerate(pbar):
            loss = self.test_func(data)
            batch_size = data[2].shape[0]
            tot_loss += loss * batch_size
            tot_cnt += batch_size
            print("test loss {}".format(tot_loss / tot_cnt) )

    def update_network(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_func(self, data):
        self.net.train()
        data, points, label = data
        data = data.cuda()
        points = points.cuda()
        label = label.cuda().view(-1, 1)
        outputs = self.net(data, points)
        loss = self.criterion(outputs, label)
        self.record_loss(loss, 'train')
        return loss

    def val_func(self, data):
        self.net.eval()
        data, points, label = data
        data = data.cuda()
        points = points.cuda()
        label = label.cuda().view(-1, 1)
        with torch.no_grad():
            outputs = self.net(data, points)
            loss = self.criterion(outputs, label)
        self.record_loss(loss, 'val')
        return loss

    def test_func(self, data):
        self.net.eval()
        data, points, label = data
        data = data.cuda()
        points = points.cuda()
        label = label.cuda().view(-1, 1)
        with torch.no_grad():
            outputs = self.net(data, points)
            loss = self.criterion(outputs, label)
        return loss

    def record_loss(self, loss, mode):
        tb = self.train_tb if mode == 'train' else self.val_tb
        tb.add_scalar('loss', loss, self.clock.step)

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Checkpoint saved at {}".format(save_path))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if isinstance(self.net, nn.DataParallel):
            torch.save({
                'clock': self.clock.make_ckpt(),
                'model_state_dict': self.net.module.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_path)
        else:
            torch.save({
                'clock': self.clock.make_ckpt(),
                'model_state_dict': self.net.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_path)
        self.net.cuda()

    def load_ckpt(self, weight_name=None):
        if weight_name == None:
            weight_name = os.listdir(self.model_dir)
            weight_name.sort()
            weight_name = weight_name[-1]
        load_path = os.path.join(self.model_dir, weight_name)
        checkpoint = torch.load(load_path)
        print("Checkpoint loaded from {}".format(load_path))
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.clock.load_ckpt(checkpoint['clock'])

    def visualization(self, number):
        self.load_ckpt("ckpt_epoch156.pth")
        name_list = json.load(open(os.path.join(self.data_path, "train_val_test_split", self.category + ".test.json"), "r"))
        random.shuffle(name_list)
        save_dir = 'visualization'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        data_path = os.path.join("processed_data", self.category)
        for i in range(number):
            file = os.path.join(data_path, name_list[i]['anno_id']+".npy")
            if not os.path.exists(file):
                continue
            data = np.load(file, allow_pickle=True)[()]
            voxel = np.array(data[64][0], dtype="float32")

            for x in range(64):
                for y in range(64):
                    for z in range(64):
                        voxel[x][y][z] = voxel[x][y][z] > 0

            vertices, triangles = libmcubes.marching_cubes(voxel, 0)
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(os.path.join(save_dir, "{}_groundtruth.obj".format(i)))

            voxel = torch.FloatTensor(voxel).unsqueeze(0).cuda()
            d_voxel = np.zeros((64, 64, 64))
            for x in range(64):
                points = []
                for y in range(64):
                    for z in range(64):
                        points.append([x, y, z])

                points = torch.FloatTensor(points).unsqueeze(0).cuda() / 64.0
                output = self.net(voxel, points)
                cnt = 0
                for y in range(64):
                    for z in range(64):
                        d_voxel[x][y][z] = output[cnt] > 0.5
                        cnt += 1
            correct = 0
            for x in range(64):
                for y in range(64):
                    for z in range(64):
                        if voxel[0][x][y][z] == d_voxel[x][y][z]:
                            correct += 1
            print("{}/{} {}".format(correct, 64 ** 3, correct / 64 **3))

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.voxels(d_voxel, facecolors='b', edgecolors='k')
            plt.savefig(os.path.join(save_dir, "{}_voxel.png".format(i)))

            vertices, triangles = libmcubes.marching_cubes(d_voxel, 0)
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(os.path.join(save_dir, "{}_result.obj".format(i)))

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

if __name__ == "__main__":
    agent = Agent("data", "Chair")
    # agent.train()
    agent.visualization(10)


