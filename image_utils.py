import cv2
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import os
from torchvision.utils import save_image
import itertools
from sklearn.cluster import SpectralClustering
import torch.nn.functional as F
import shutil

def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def color2gray(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_img_3d = image_array.copy()
    gray_img_3d[:, :, 0] = gray
    gray_img_3d[:, :, 1] = gray
    gray_img_3d[:, :, 2] = gray
    return gray_img_3d

def crop(image_array):
    image = cv2.resize(image_array, (128, 128))
    w = np.random.randint(0, 16)
    h = np.random.randint(0, 16)
    image = image[w:w+112,h:h+112,:]
    return image

def rotation(image_array):
    h,w = image_array.shape[:2]
    center = (w//2, h//2)
    angle = np.random.randint(-30, 30)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image_array, M, (w, h))
    return rotated


def _meshgrid(height, width, nb):
    x_t = torch.matmul(torch.ones(size = (height, 1)),
                            torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 0)).cuda()  ###[-1,0, 1],[-1, 0, 1]]
    y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1),
                            torch.ones(size=(1, width))).cuda()  ###[-1,-1, -1], [0, 0, 0], [1,1,1]]

    x_t = torch.unsqueeze(x_t, 0).repeat(nb, 1, 1)
    y_t = torch.unsqueeze(y_t, 0).repeat(nb, 1, 1)
    return x_t, y_t


def _repeat(x, n_repeats):
    rep = torch.ones([1, n_repeats])  ###[1, 64 * 64]
    rep = rep.long()
    x = torch.matmul(torch.reshape(x, (-1, 1)), rep).cuda()  ###[batch, 64 * 64]
    return torch.reshape(x, [-1])


def _interpolate(im, x, y, xd, yd, out_size):
    num_batch, channels, height, width = im.shape
    x = x.float()
    y = y.float()
    xd = xd.float()
    yd = yd.float()
    height_f = np.float(height)  ##64
    width_f = np.float(width)
    out_height = out_size[0]  ##64
    out_width = out_size[1]

    zero1 = np.zeros(shape = [])
    max_y1 = np.float(im.shape[3] -1)
    max_x1 = np.float(im.shape[2] -1)

    # scale indices from [-1, 1] to [0, width/height]
    x = (x + 1.0) * (width_f) / 2.0
    y = (y + 1.0) * (height_f) / 2.0
    ###################
    x = x + xd  ###
    y = y + yd

    x = torch.clamp(x, zero1 + 0.00001, max_x1 - 0.00001)  ##[409600] 0-62.....
    y = torch.clamp(y, zero1 + 0.00001, max_y1 - 0.00001)
    ##################
    # do sampling
    x0 = x.floor() ###[409600]
    x1 = x0 + 1
    y0 = y.floor()
    y1 = y0 + 1

    dim2 = width
    dim1 = width * height
    base = _repeat(torch.arange(num_batch) * dim1,
                   out_height * out_width).cuda()  ###batch * 64 * 64 [[0]*64*64, [1]*64*64,....]

    base_y0 = base + y0 * dim2 
    base_y1 = base + y1 * dim2
    idx_a = (base_y0 + x0).int()
    idx_b = (base_y1 + x0).int()
    idx_c = (base_y0 + x1).int()
    idx_d = (base_y1 + x1).int()

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = torch.reshape(im, (-1, channels))  ##[100, 64, 64, 3]  [409600, 3]
    im_flat = im_flat.int()

    Ia = torch.index_select(im_flat, 0,  idx_a)
    Ib = torch.index_select(im_flat, 0, idx_b)  ###indices[408676] = 409644 is not in [0, 409600)
    Ic = torch.index_select(im_flat, 0, idx_c)
    Id = torch.index_select(im_flat, 0, idx_d)

    # and finally calculate interpolated values
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()
    wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
    wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
    wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
    wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
    output = wa * Ia + wb * Ib + wc * Ic +wd * Id  
    return output, x, y


def warpnn(im, dlm, nb):  ###dlm: [64, 2 50176]
    imgsize, nc = im.shape[2], im.shape[1]
    grid_x, grid_y = _meshgrid(imgsize, imgsize, nb) ##[b, imgsize, img_size] (-1, 1)
    grid_x, grid_y = (grid_x + 1.0) * imgsize // 2,  (grid_y + 1.0) * imgsize // 2
    x_s, y_s = dlm[:, 0, :, :], dlm[:, 1, :, :]
    coord = [x_s, y_s]
    x, y = grid_x + x_s, grid_y + y_s
    x, y = torch.reshape(x, [-1]), torch.reshape(y, [-1])

    x = torch.clamp(x, 0 + 0.00001, imgsize - 1 - 0.00001)
    y = torch.clamp(y, 0 + 0.00001, imgsize - 1 - 0.00001)
    x0 = x.floor()
    x1 = x0 + 1
    y0 = y.floor()
    y1 = y0 + 1

    im = torch.reshape(im.permute(0, 2, 3, 1), (-1, nc))
    base_b = _repeat(torch.arange(nb) * imgsize * imgsize, imgsize * imgsize)
    idx_a = (base_b + y0 * imgsize + x0).int()
    idx_b = (base_b + y0 * imgsize + x1).int()
    idx_c = (base_b + y1 * imgsize + x0).int()
    idx_d = (base_b + y1 * imgsize + x1).int()
    Ia = torch.index_select(im, 0, idx_a)   ##[b * imgsize * imgsize, 3]
    Ib = torch.index_select(im, 0, idx_b)
    Ic = torch.index_select(im, 0, idx_c)
    Id = torch.index_select(im, 0, idx_d)

    wa = torch.unsqueeze((x1 - x) * (y1 - y), 1)
    wb = torch.unsqueeze((x - x0) * (y1 - y), 1)
    wc = torch.unsqueeze((x1 - x) * (y - y0), 1)
    wd = torch.unsqueeze((x - x0) * (y - y0), 1)
    output = wa * Ia + wb * Ib + wc * Ic + wd * Id
    #output = output.transpose(1, 0)
    output = torch.reshape(output, (nb, imgsize, imgsize, nc))
    output = output.permute(0, 3, 1, 2)
    return output, coord


def plot_grid(grid, appx, gx, images, labels, model_path):  ##[batch, 2, 64, 64]
    model_path = model_path + '_image/'
    batch = labels.shape[0]
    imgsize = images.shape[2]
    dx, dy = grid[0].cpu().numpy() , grid[1].cpu().numpy()
    # print(np.min(dx), np.max(dx), np.min(dy), np.max(dy))
    u, v = np.arange(imgsize), np.arange(imgsize)
    u,v = np.meshgrid(u, v)
    # v = v[::-1, :]  ##deal with coordinates difference 0->111
    # u = u[:, ::-1]
    for idx in range(batch):
        label = labels[idx].item()
        # save_path = os.path.join(os.getcwd(), '1/%s' % str(label))
        # save_g = os.path.join(save_path, 'geo')
        # if not os.path.exists(save_g):
        #     os.makedirs(save_g)
        plt.axis('off')
        # for i in range(imgsize):
        #     for j in range(imgsize):
        #         if dx[idx, i, j] < 0.6 and dx[idx, i, j] > -0.6:
        #             dx[idx, i, j] = 0
        #
        #         if dy[idx, i, j] < 0.6 and dy[idx, i, j] > -0.6:
        #             dy[idx, i, j] = 0
        # plt.quiver(dx[idx, ::2, ::2], dy[idx, ::2, ::2], scale = 30)
        # plt.savefig(os.path.join(save_g, str(len(os.listdir(save_g))) + '.png'), bbox_inches='tight', pad_inches=0)
        # plt.close()

        # norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        # plt.imshow(X[idx], cmap='seismic', interpolation='bicubic', norm = norm)
        # plt.savefig(os.path.join(save_g, str(len(os.listdir(save_g))) + '.png'), bbox_inches='tight', pad_inches=0)
        # plt.close()
        #
        save_path = os.path.join(os.getcwd(), model_path + str(label))
        x = dx[idx] + u
        y = dy[idx] + v
        # y = y[::-1, :]
        x_ = x.transpose(1, 0)
        y_ = y.transpose(1, 0)
        plt.plot(x[::2, ::2], y[::2, ::2], linewidth=1, color='blue')
        plt.plot(x_[::2, ::2], y_[::2,::2], linewidth=1, color='blue')
        # plt.quiver(x[::2, ::2], y[::2, ::2], scale=20)
        save_g = os.path.join(save_path, 'geo')
        if not os.path.exists(save_g):
            os.makedirs(save_g)
        plt.savefig(os.path.join(save_g, str(idx) + '.png'), bbox_inches='tight', pad_inches=0)
        # plt.close()
        save_ag = os.path.join(save_path, 'ag')
        if not os.path.exists(save_ag):
            os.makedirs(save_ag)
        img = np.transpose(gx[idx],(1, 2 , 0))
        plt.imshow(img)
        plt.savefig(os.path.join(save_ag, str(idx) + '.png'))
        plt.close()
        # ##save app
        # save_app = os.path.join(save_path, 'app')
        # if not os.path.exists(save_app):
        #    os.makedirs(save_app)
        # save_name = os.path.join(save_app, str(len(os.listdir(save_app))) + '.png')
        # save_image(appx[idx], save_name, nrow=1, normalize=True)
    #     #
    #     #
    #     # # ###save image
        save_img = os.path.join(save_path, 'img')
        if not os.path.exists(save_img):
            os.makedirs(save_img)
        save_name = os.path.join(save_img, str(idx) + '.png')
        save_image(images[idx], save_name , nrow=1, normalize=True)
    #
        save_img = os.path.join(save_path, 'gx')
        if not os.path.exists(save_img):
           os.makedirs(save_img)
        save_name = os.path.join(save_img, str(idx) + '.png')
        save_image(gx[idx], save_name , nrow=1, normalize=True)


