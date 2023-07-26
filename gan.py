"""
#Deep Learning project

##MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation

In this notebook we demonstrated the use of a Convolutional Generative Adversarial Network to generate music with 3 Insruments, Piano, Guitar and Bass.
This code is based on the code presented by the paper "MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation". In the paper, the authors generated music for Piano only. Our code crate a model to generate music for 2 additional instruments.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from common import *


def load_array_from_json_element(file_content):
    js = json.loads(file_content)
    instruments = js["Notes"]
    notes = np.array(js["Instruments"])

    # find the instruments in the sample
    groups = []
    piano_count = 0
    for inst in instruments:
        group = get_midi_group(inst)
        if group == 0:
            groups.append(piano_count)
        elif group < 8:
            groups.append(group + 3)
        elif group < 40:
            groups.append(group - 16 + 3)

    # reshape array
    num_mini_arrays = notes.shape[1] // 32
    mini_arrays = notes[:, :num_mini_arrays * 32, :].reshape(notes.shape[0], num_mini_arrays, 32, notes.shape[2])

    ## 3 rows for piano and all the groups piano guitar and bass

    list = [np.zeros((mini_arrays.shape[1], mini_arrays.shape[2], mini_arrays.shape[3]))] * 27
    ## assign notes to the right place
    instrument_count = 0
    for i in range(len(groups)):
        list[i] = mini_arrays[instrument_count]
        instrument_count = instrument_count + 1
    bars = np.array(list)
    bars = np.transpose(bars, (1, 0, 2, 3))

    return bars

def load_array_from_json_element_3D_only(file_content):
    js = json.loads(file_content)
    instruments = js["Notes"]
    notes = np.array(js["Instruments"])

    res = np.zeros((3, notes.shape[1], 128))

    # find the instruments in the sample
    inst_counter = 0
    for inst in instruments:
        group = get_midi_group(inst)
        if group < 8:
            res[0] = notes[inst_counter]
        elif 24 < group < 32:
            res[1] = notes[inst_counter]
        elif group < 40:
            res[2] = notes[inst_counter]
            inst_counter = inst_counter + 1

        # reshape array
    num_mini_arrays = res.shape[1] // 32
    mini_arrays = res[:, :num_mini_arrays * 32, :].reshape(res.shape[0], num_mini_arrays, 32, notes.shape[2])

    bars = np.transpose(mini_arrays, (1, 0, 2, 3))
    return bars

def load_from_json_elements(directory=part_dir, max_files_to_load=30, current_file_index=0):
    ## Training on 30 files each time due to RAM capabilities
    x = []
    files_counter = 1
    file_n = current_file_index
    y = glob.glob(os.path.join(directory, '*'))

    for file_n, f in enumerate(y):
        if file_n < current_file_index + max_files_to_load:
            if files_counter <= max_files_to_load:
                print("loading file: " + f)
                if os.path.isfile(f):
                    with gzip.open(f, 'rb') as f:
                        x.append(load_array_from_json_element_3D_only(f.read()))
                        files_counter = files_counter + 1
    X = np.concatenate(x)
    return X, current_file_index


"""##Operational Functions"""


class ops:
    @staticmethod
    def conv_cond_concat(x, y):
        """Concatenate conditioning vector on feature map axis."""
        x_shapes = x.shape
        y_shapes = y.shape
        y2 = y.expand(x_shapes[0], y_shapes[1], x_shapes[2], x_shapes[3])

        return torch.cat((x, y2), 1)

    @staticmethod
    def conv_prev_concat(x, y):
        """Concatenate conditioning vector on feature map axis."""
        x_shapes = x.shape
        y_shapes = y.shape
        if x_shapes[2:] == y_shapes[2:]:
            y2 = y.expand(x_shapes[0], y_shapes[1], x_shapes[2], x_shapes[3], x_shapes[4])

            return torch.cat((x, y2), 1)

        else:
            print(x_shapes[2:])
            print(y_shapes[2:])

    @staticmethod
    def batch_norm_1d(x):
        x_shape = x.shape[1]
        batch_nor = nn.BatchNorm1d(x_shape, eps=1e-05, momentum=0.9, affine=True)
        batch_nor = batch_nor.cuda()

        output = batch_nor(x)
        return output

    @staticmethod
    def batch_norm_1d_cpu(x):
        x_shape = x.shape[1]
        # ipdb.set_trace()
        # batch_nor = nn.BatchNorm1d(x_shape, eps=1e-05, momentum=0.9, affine=True)
        # output = batch_nor(x)
        output = x
        return output

    @staticmethod
    def batch_norm_2d(x):
        x_shape = x.shape[1]
        batch_nor = nn.BatchNorm2d(x_shape, eps=1e-05, momentum=0.9, affine=True)
        batch_nor = batch_nor.cuda()
        output = batch_nor(x)
        return output

    @staticmethod
    def batch_norm_3d(x):
        x_shape = x.shape[1]
        batch_nor = nn.BatchNorm3d(x_shape, eps=1e-05, momentum=0.9, affine=True)
        batch_nor = batch_nor.cuda()
        output = batch_nor(x)
        return output

    @staticmethod
    def batch_norm_3d_cpu(x):
        output = x
        return output

    @staticmethod
    def sigmoid_cross_entropy_with_logits(inputs, labels):
        loss = nn.BCEWithLogitsLoss()
        output = loss(inputs, labels)
        return output

    @staticmethod
    def reduce_mean(x):
        output = torch.mean(x, 0, keepdim=False)
        output = torch.mean(output, -1, keepdim=False)
        return output

    @staticmethod
    def reduce_mean_0(x):
        output = torch.mean(x, 0, keepdim=False)
        return output

    @staticmethod
    def l2_loss(x, y):
        loss_ = nn.MSELoss(reduction='sum')
        l2_loss_ = loss_(x, y) / 2
        return l2_loss_

    @staticmethod
    def lrelu(x, leak=0.2):
        z = torch.mul(x, leak)
        return torch.max(x, z)


"""##The Model"""


class generator(nn.Module):
    def __init__(self,pitch_range):
        super(generator, self).__init__()
        self.n_instruments = 3
        self.pitch_range = 128
        self.gf_dim=64

        self.h1  = nn.ConvTranspose3d(in_channels=160, out_channels=self.pitch_range, kernel_size=(1,2,1), stride=(1,2,2))
        self.h2  = nn.ConvTranspose3d(in_channels=160, out_channels=self.pitch_range, kernel_size=(1,2,1), stride=(1,2,2))
        self.h3  = nn.ConvTranspose3d(in_channels=160, out_channels=self.pitch_range, kernel_size=(1,2,1), stride=(1,2,2))
        self.h4  = nn.ConvTranspose3d(in_channels=160, out_channels=1, kernel_size=(1,2, self.pitch_range), stride=(1,2,2))

        self.h0_prev = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(1,1,pitch_range), stride=(1,2,2))
        self.h1_prev = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(1,2,1), stride=(1,2,2))
        self.h2_prev = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(1,2,1), stride=(1,2,2))
        self.h3_prev = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(1,2,1), stride=(1,2,2))

        self.linear1 = nn.Linear(100,1024)
        self.linear2 = nn.Linear(1024,self.gf_dim*self.n_instruments*2*2*1)

    def forward(self, z, prev_x, y ,batch_size,pitch_range):
        prev_x = prev_x.reshape((prev_x.shape[0],1,prev_x.shape[1],prev_x.shape[2],prev_x.shape[3]))

        h0_prev  = self.h0_prev(prev_x)
        h0_prev = ops.lrelu(ops.batch_norm_3d(h0_prev),0.2)
        h1_prev = self.h1_prev(h0_prev)

        h1_prev = ops.lrelu(ops.batch_norm_3d(h1_prev),0.2)
        h2_prev = ops.lrelu(ops.batch_norm_3d(self.h2_prev(h1_prev)),0.2)

        h3_prev = ops.lrelu(ops.batch_norm_3d(self.h3_prev(h2_prev)),0.2)

        h0 = F.relu(ops.batch_norm_1d(self.linear1(z)))

        h1 = F.relu(ops.batch_norm_1d(self.linear2(h0)))
        h1 = h1.view(batch_size, self.gf_dim * 2,self.n_instruments, 2, 1)
        h1 = ops.conv_prev_concat(h1,h3_prev)

        h2 = F.relu(ops.batch_norm_3d(self.h1(h1)))
        h2 = ops.conv_prev_concat(h2,h2_prev)


        h3 = F.relu(ops.batch_norm_3d(self.h2(h2)))
        h3 = ops.conv_prev_concat(h3,h1_prev)

        h4 = F.relu(ops.batch_norm_3d(self.h3(h3)))
        h4 = ops.conv_prev_concat(h4,h0_prev)

        g_x = torch.sigmoid(self.h4(h4))
        g_x = g_x.reshape((g_x.shape[0],g_x.shape[2],g_x.shape[3],g_x.shape[4]))
        return g_x

class sample_generator(nn.Module):
    def __init__(self):
        super(sample_generator, self).__init__()
        self.n_instruments = 3
        self.pitch_range = 128
        self.gf_dim=64

        self.h1  = nn.ConvTranspose3d(in_channels=160, out_channels=self.pitch_range, kernel_size=(1,2,1), stride=(1,2,2))
        self.h2  = nn.ConvTranspose3d(in_channels=160, out_channels=self.pitch_range, kernel_size=(1,2,1), stride=(1,2,2))
        self.h3  = nn.ConvTranspose3d(in_channels=160, out_channels=self.pitch_range, kernel_size=(1,2,1), stride=(1,2,2))
        self.h4  = nn.ConvTranspose3d(in_channels=160, out_channels=1, kernel_size=(1,2, self.pitch_range), stride=(1,2,2))

        self.h0_prev = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(1,1, self.pitch_range), stride=(1,2,2))
        self.h1_prev = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(1,2,1), stride=(1,2,2))
        self.h2_prev = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(1,2,1), stride=(1,2,2))
        self.h3_prev = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(1,2,1), stride=(1,2,2))


        self.linear1 = nn.Linear(100,1024)
        self.linear2 = nn.Linear(1024,self.gf_dim*self.n_instruments*2*2*1)


    def forward(self, z, prev_x, y ,batch_size,pitch_range):

        prev_x = prev_x.reshape((prev_x.shape[0],1,prev_x.shape[1],prev_x.shape[2],prev_x.shape[3]))

        h0_prev  = self.h0_prev(prev_x)
        h0_prev = ops.lrelu(ops.batch_norm_3d_cpu(h0_prev),0.2)
        h1_prev = self.h1_prev(h0_prev)

        h1_prev = ops.lrelu(ops.batch_norm_3d_cpu(h1_prev),0.2)
        h2_prev = ops.lrelu(ops.batch_norm_3d_cpu(self.h2_prev(h1_prev)),0.2)

        h3_prev = ops.lrelu(ops.batch_norm_3d_cpu(self.h3_prev(h2_prev)),0.2)

        h0 = F.relu(ops.batch_norm_1d_cpu(self.linear1(z)))

        h1 = F.relu(ops.batch_norm_1d_cpu(self.linear2(h0)))
        h1 = h1.view(batch_size, self.gf_dim * 2,self.n_instruments, 2, 1)
        h1 = ops.conv_prev_concat(h1,h3_prev)

        h2 = F.relu(ops.batch_norm_3d_cpu(self.h1(h1)))
        h2 = ops.conv_prev_concat(h2,h2_prev)


        h3 = F.relu(ops.batch_norm_3d_cpu(self.h2(h2)))
        h3 = ops.conv_prev_concat(h3,h1_prev)

        h4 = F.relu(ops.batch_norm_3d_cpu(self.h3(h3)))
        h4 = ops.conv_prev_concat(h4,h0_prev)

        g_x = torch.sigmoid(self.h4(h4))
        g_x = g_x.reshape((g_x.shape[0],g_x.shape[2],g_x.shape[3],g_x.shape[4]))
        return g_x



class discriminator(nn.Module):
    def __init__(self,pitch_range):
        super(discriminator, self).__init__()
        self.dfc_dim = 64

        self.h0_prev = nn.Conv3d(1,32, kernel_size=(2,2,2),stride=(2, 2, 2))
        self.h1_prev = nn.Conv3d(32,32, kernel_size=(1,2,1), stride=(2, 2, 2))

        self.linear0 =  nn.Linear(8192,1280)
        self.linear1 = nn.Linear(1280,1024)
        self.linear2 = nn.Linear(1024,1)


    def forward(self,x,y,batch_size,pitch_range):
        x = x.reshape((x.shape[0],1,x.shape[1],x.shape[2],x.shape[3]))
        h0 = ops.lrelu(self.h0_prev(x),0.2)
        fm = h0

        h1 = self.h1_prev(h0)
        h1 = ops.lrelu(ops.batch_norm_3d(h1),0.2)
        h1 = h1.view(batch_size, -1)
        h1 = ops.lrelu(ops.batch_norm_1d(self.linear0(h1)))

        h2 = ops.lrelu(ops.batch_norm_1d(self.linear1(h1)))

        h3 = self.linear2(h2)
        h3_sigmoid = torch.sigmoid(h3)


        return h3_sigmoid, h3, fm

class get_dataloader(object):
    def __init__(self, data, prev_data, y=None):
        self.size = data.shape[0]
        self.data = torch.from_numpy(data).float()
        self.prev_data = torch.from_numpy(prev_data).float()
        # self.y   = torch.from_numpy(y).float()

        # self.label = np.array(label)

    def __getitem__(self, index):
        return self.data[index], self.prev_data[index]  # , self.y[index]

    def __len__(self):
        return self.size


def load_data(X):
    #######load the data########
    check_range_st = 0
    check_range_ed = 129
    X_tr = X[1:X.shape[0] - 1, :, :, :]
    prev_X_tr = X[0:X.shape[0] - 2, :, :, :]
    X_tr = X_tr[:, :, :, check_range_st:check_range_ed]
    prev_X_tr = prev_X_tr[:, :, :, check_range_st:check_range_ed]

    train_iter = get_dataloader(X_tr, prev_X_tr)
    kwargs = {'num_workers': 4, 'pin_memory': True}  # if args.cuda else {}
    train_loader = DataLoader(train_iter, batch_size=72, shuffle=True, **kwargs)
    print('data preparation is completed')
    #######################################
    return train_loader


def main(X, big_epoch):
    global netG
    global netD

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 20
    lr = 0.0002

    check_range_st = 0
    check_range_ed = 129
    pitch_range = check_range_ed - check_range_st - 1

    train_loader = load_data(X)

    if not netG:
        netG = generator(pitch_range).to(device)
        netD = discriminator(pitch_range).to(device)
        netD.train()
        netG.train()

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    nz = 100
    real_label = 1
    fake_label = 0
    lossD_list = []
    lossD_list_all = []
    lossG_list = []
    lossG_list_all = []
    D_x_list = []
    D_G_z_list = []
    for epoch in range(epochs):
        sum_lossD = 0
        sum_lossG = 0
        sum_D_x = 0
        sum_D_G_z = 0
        for i, (data, prev_data) in enumerate(train_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data.to(device)
            prev_data_cpu = prev_data.to(device)
            # chord_cpu = chord.to(device)

            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)
            D, D_logits, fm = netD(real_cpu, None, batch_size, pitch_range)

            #####loss
            d_loss_real = ops.reduce_mean(ops.sigmoid_cross_entropy_with_logits(D_logits, 0.9 * torch.ones_like(D)))
            d_loss_real.backward(retain_graph=True)
            D_x = D.mean().item()
            sum_D_x += D_x

            # train with fake
            noise = torch.randn(batch_size, nz, device=device)

            fake = netG(noise, prev_data_cpu, None, batch_size, pitch_range)
            label.fill_(fake_label)
            D_, D_logits_, fm_ = netD(fake.detach(), None, batch_size, pitch_range)
            d_loss_fake = ops.reduce_mean(ops.sigmoid_cross_entropy_with_logits(D_logits_, torch.zeros_like(D_)))

            d_loss_fake.backward(retain_graph=True)
            D_G_z1 = D_.mean().item()
            errD = d_loss_real + d_loss_fake
            errD = errD.item()
            lossD_list_all.append(errD)
            sum_lossD += errD
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            D_, D_logits_, fm_ = netD(fake, None, batch_size, pitch_range)

            ###loss
            g_loss0 = ops.reduce_mean(ops.sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_)))
            # Feature Matching
            features_from_g = ops.reduce_mean_0(fm_)
            features_from_i = ops.reduce_mean_0(fm)
            fm_g_loss1 = torch.mul(ops.l2_loss(features_from_g, features_from_i), 0.1)

            mean_image_from_g = ops.reduce_mean_0(fake)
            smean_image_from_i = ops.reduce_mean_0(real_cpu)
            fm_g_loss2 = torch.mul(ops.l2_loss(mean_image_from_g, smean_image_from_i), 0.01)

            errG = g_loss0 + fm_g_loss1 + fm_g_loss2
            errG.backward(retain_graph=True)
            D_G_z2 = D_.mean().item()
            optimizerG.step()

            ############################
            # (3) Update G network again: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            D_, D_logits_, fm_ = netD(fake, None, batch_size, pitch_range)

            ###loss
            g_loss0 = ops.reduce_mean(ops.sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_)))
            # Feature Matching
            features_from_g = ops.reduce_mean_0(fm_)
            features_from_i = ops.reduce_mean_0(fm)
            loss_ = nn.MSELoss(reduction='sum')
            feature_l2_loss = loss_(features_from_g, features_from_i) / 2
            fm_g_loss1 = torch.mul(feature_l2_loss, 0.1)

            mean_image_from_g = ops.reduce_mean_0(fake)
            smean_image_from_i = ops.reduce_mean_0(real_cpu)
            mean_l2_loss = loss_(mean_image_from_g, smean_image_from_i) / 2
            fm_g_loss2 = torch.mul(mean_l2_loss, 0.01)
            errG = g_loss0 + fm_g_loss1 + fm_g_loss2
            sum_lossG += errG
            errG.backward()
            lossG_list_all.append(errG.item())

            D_G_z2 = D_.mean().item()
            sum_D_G_z += D_G_z2
            optimizerG.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f', errD, errG, D_x,
                  D_G_z1, D_G_z2)

            average_lossD = sum_lossD / len(train_loader.dataset)
            average_lossG = sum_lossG / len(train_loader.dataset)
            average_D_x = sum_D_x / len(train_loader.dataset)
            average_D_G_z = sum_D_G_z / len(train_loader.dataset)

            lossD_list.append(average_lossD)
            lossG_list.append(average_lossG)
            D_x_list.append(average_D_x)
            D_G_z_list.append(average_D_G_z)

            print(
                '==> Epoch: {} Average lossD: {:.10f} average_lossG: {:.10f},average D(x): {:.10f},average D(G(z)): {:.10f} '.format(
                    epoch, average_lossD, average_lossG, average_D_x, average_D_G_z))

            # do checkpointing
            torch.save(netG.state_dict(),
                       '%s/netG_epoch_%d.pth' % (out_dir, big_epoch))
            torch.save(netD.state_dict(),
                       '%s/netD_epoch_%d.pth' % (out_dir, big_epoch))


def run_training():
    global netG
    global netD

    for j in [0, 2]:
        current_file_index = 0
        for i in range(20):
            print("Start Training on batch " + str(i))
            X, current_file_index = load_from_json_elements(current_file_index=current_file_index,
                                                            directory=os.path.join(full_dir, str(j)))
            main(X, i)


"""###Sampeling"""


def sample():
    global netG
    global netD
    batch_size = 1
    nz = 100
    n_bars = 20
    X, i = load_from_json_elements(directory=part_dir, current_file_index=0)
    X_te = X[20:21, :, :, :]
    prev_X_te = X[21:22, :, :, :]

    test_iter = get_dataloader(X_te, prev_X_te)  # get_dataloader(X_te,prev_X_te,y_te)
    kwargs = {'num_workers': 4, 'pin_memory': True}  # if args.cuda else {}
    test_loader = DataLoader(test_iter, batch_size=batch_size, shuffle=False, **kwargs)

    if not netG:
        netG = sample_generator()
        netG.load_state_dict(torch.load(os.path.join(out_dir, 'netG_epoch_19.pth')))
        netG.eval()
    output_songs = []
    for i, (data, prev_data) in enumerate(test_loader, 0):
        list_song = []
        first_bar = data[0].view(1, 3, 32, 128)
        list_song.append(first_bar)

        noise = torch.randn(n_bars, nz)

        for bar in range(n_bars):
            z = noise[bar].view(1, nz)
            if bar == 0:
                prev = data[0].view(1, 3, 32, 128)
                pass
            else:
                prev = list_song[bar - 1].view(1, 3, 32, 128)
            sample = torch.round(netG(z, prev, None, batch_size, 128))
            list_song.append(sample)
        output_songs.append(list_song)

        print('num of output_songs: {}'.format(len(output_songs)))

    print('creation completed, check out what I make!')
    return output_songs


"""###Generate Midi File"""


def generate_midi(output_songs):
  instruments = []
  midi_maps = []

  for songs in output_songs:
    song = songs[0]
    for bar in songs:

      song = torch.cat((song,bar),0)

  print(song.shape)

  song = torch.transpose(song,1,0).reshape(3, (len(output_songs[0])+1)*32, 128)

  for inst in range(3):
    if inst < 3:
      if not torch.all(song[inst] == 0):
        if inst == 0:
          instruments.append(0)
        if inst == 1:
          instruments.append(26)
        if inst == 2:
          instruments.append(33)
        midi_maps.append(song[inst].tolist())

  print(len(midi_maps))
  write_midi_maps(midi_maps,instruments)


if __name__ == '__main__':
    netG = None
    netD = None
    run_training()
    generate_midi(sample())
