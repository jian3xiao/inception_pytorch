import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from scipy import linalg
import warnings
from torchvision import datasets, transforms
from torchsummary import summary
# from torchvision.models.inception import inception_v3
from metric.inception_3 import inception_v3

import numpy as np
from glob import glob
import os
from PIL import Image


def get_images(filename):
    x = Image.open(filename).convert('RGB')
    # x = x.resize((299, 299), resample=Image.BILINEAR)
    x_arr = np.array(x)
    # # x_arr = (x_arr - 127.5) / 127.5  # [-1, 1]
    x_arr = np.transpose(x_arr, axes=[2, 0, 1])
    return x_arr


def trans(batch_tensor):
    x_ch0 = (torch.unsqueeze(batch_tensor[:, 0], 1) - 127.5) / 127.5
    x_ch1 = (torch.unsqueeze(batch_tensor[:, 1], 1) - 127.5) / 127.5
    x_ch2 = (torch.unsqueeze(batch_tensor[:, 2], 1) - 127.5) / 127.5
    batch_tensor = torch.cat((x_ch0, x_ch1, x_ch2), 1)
    return batch_tensor


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-8):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


device = torch.device("cuda:0")
dtype = torch.cuda.FloatTensor

filenames_fake = glob(os.path.join('', '*.*'))
# filenames_fake = sorted(filenames_fake)
# images = [get_images(filename) for filename in filenames]  # h, w, c  [0:10000]
num = len(filenames_fake)
print('image numbers:', num)

filenames_real = glob(os.path.join('', '*.*'))
# filenames_real = sorted(filenames_real)
# images_real = [get_images(filename) for filename in filenames_real]

# finetune model
model_ft = inception_v3(pretrained=True, transform_input=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200)  # 200, 196, 120
model_ft = model_ft.to(device)
print('-' * 30)
# summary(model_ft, (3, 299, 299))

# load param
model_ft.load_state_dict(torch.load('.pth'))
model_ft.eval()

splits = 5
bs = 100 
preds_f = []
preds_r = []
pool_f = []
pool_r = []
n_batches = num // bs
up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).type(dtype)

for i in range(n_batches):
    # load fake images
    batch_list = filenames_fake[(i * bs):min((i + 1) * bs, num)]
    imgs = [get_images(filename) for filename in batch_list]
    batch_imgs = np.array(imgs)  # (bs, 3, h, w)
    batch_imgs = torch.from_numpy(batch_imgs).to(device).type(dtype)

    batch_imgs = up(batch_imgs)
    batch_imgs = trans(batch_imgs)

    # load real images
    batch_list_real = filenames_real[(i * bs):min((i + 1) * bs, num)]
    imgs_real = [get_images(filename) for filename in batch_list_real]
    batch_imgs_real = np.array(imgs_real)
    batch_imgs_real = torch.from_numpy(batch_imgs_real).to(device).type(dtype)

    batch_imgs_real = up(batch_imgs_real)
    batch_imgs_real = trans(batch_imgs_real)

    # fake logits and pools for IS
    with torch.no_grad():  # 使用该句不影响最终数值，但会是显存占有减少
        pred, pool3 = model_ft(batch_imgs)  # [bs, 200]  out是softmax激活前的输入，即logits
        logits = torch.nn.functional.softmax(pred, dim=1)  # softmax转换成概率
    pool3 = pool3.view(bs, -1)
    preds_f.append(logits.data.cpu().numpy())
    pool_f.append(pool3.data.cpu().numpy())

    # real logits and pool3 for IS
    with torch.no_grad():
        _, pool3_r = model_ft(batch_imgs_real)  # pred_r  [bs, 200]  out是softmax激活前的输入，即logits
    # logits_r = torch.nn.functional.softmax(pred_r, dim=1)  # softmax转换成概率
    pool3_r = pool3_r.view(bs, -1)
    # preds_r.append(logits_r.data.cpu().numpy())
    pool_r.append(pool3_r.data.cpu().numpy())

# IS via fake logits
preds_fake = np.concatenate(preds_f, 0)  # [10000, 1000] np
scores_f = []
for i in range(splits):
    part = preds_fake[(i * preds_fake.shape[0] // splits):((i + 1) * preds_fake.shape[0] // splits), :]
    kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    scores_f.append(np.exp(kl))
print('fake image IS:', np.mean(scores_f), np.std(scores_f))  #

# IS via real logits
# preds_real = np.concatenate(preds_r, 0)  # [10000, 1000] np
# scores_r = []
# for i in range(splits):
#     part = preds_real[(i * preds_real.shape[0] // splits):((i + 1) * preds_real.shape[0] // splits), :]
#     kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
#     kl = np.mean(np.sum(kl, 1))
#     scores_r.append(np.exp(kl))
# print('real image IS:', np.mean(scores_r), np.std(scores_r))  #

# FID via fake pool3 and real pool3
act_f = np.concatenate(pool_f, 0)  # (num, 2048)
act_r = np.concatenate(pool_r, 0)
mu1 = np.mean(act_f, axis=0)  # 2048
sigma1 = np.cov(act_f, rowvar=False)
mu2 = np.mean(act_r, axis=0)
sigma2 = np.cov(act_r, rowvar=False)  # (2048, 2048)
fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
print('FID:', fid_value)

