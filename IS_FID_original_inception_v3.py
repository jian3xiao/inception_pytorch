import torch
from torch import nn
import torch.utils.data as data
from torchvision import datasets, transforms
# from torchvision.models.inception import inception_v3
from metric.inception_3 import inception_v3  # 与上面的相同，只不过增加了pool3层的输出
import numpy as np
from scipy import linalg
import warnings
import random


manualSeed = 999  # manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

bs = 200
splits = 5
class_num = 200  # 200, 196, 120
device = torch.device("cuda:0")

transform = transforms.Compose([transforms.Resize((299, 299)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] .  (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

real_dir = ''
real_dataset = datasets.ImageFolder(real_dir, transform)
real_dataloaders = torch.utils.data.DataLoader(real_dataset, batch_size=bs, shuffle=Ture, num_workers=4)
print('real images number:', len(real_dataset))

fake_dir = ''
fake_dataset = datasets.ImageFolder(fake_dir, transform)
fake_dataloaders = torch.utils.data.DataLoader(fake_dataset, batch_size=bs, shuffle=Ture, num_workers=4)
print('fake images number:', len(fake_dataset))

# ###################### 模型加载 #########################
# 使用原本的（非微调的）inception_v3的参数
model = inception_v3(pretrained=True, transform_input=False)
model.to(device)
model.eval()

# finetune model
# model = inception_v3(pretrained=True, transform_input=False)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, class_num)  # 200, 196, 120
# model = model.to(device)
# model.load_state_dict(torch.load(' .pth'))
# model.eval()

# ########### real images ####################
r_preds = []
r_pools = []
for data, _ in real_dataloaders:  # 第二个是label，因为只有一个文件夹，所以label都是0
    r_data = data.to(device)
    with torch.no_grad():  # 使用该句不影响最终数值，但会是显存占有减少
        output, pool3 = model(r_data)  # 是softmax激活前的输入，即logits. [bs, 1000]
        pred = torch.nn.functional.softmax(output, dim=1)
        r_preds.append(pred.cpu())  # list

        pool3 = pool3.view(bs, -1)
        r_pools.append(pool3.cpu())
r_pred = np.concatenate(r_preds, 0)  # (30000, 1000)
r_act = np.concatenate(r_pools, 0)  # (30000, 2048)

# Now compute the IS
r_scores = []
for i in range(splits):
    part = r_pred[(i * r_pred.shape[0] // splits):((i + 1) * r_pred.shape[0] // splits), :]
    kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    r_scores.append(np.exp(kl))
print('real image IS:', np.mean(r_scores), np.std(r_scores))

# ########### fake images #############################
f_preds = []
f_pools = []
for data, _ in fake_dataloaders:
    f_data = data.to(device)
    with torch.no_grad():
        output, pool3 = model(f_data)
        pred = torch.nn.functional.softmax(output, dim=1)
        f_preds.append(pred.cpu())  # list

        pool3 = pool3.view(bs, -1)
        f_pools.append(pool3.cpu())
f_pred = np.concatenate(f_preds, 0)
f_act = np.concatenate(f_pools, 0)

f_scores = []
for i in range(splits):
    part = f_pred[(i * f_pred.shape[0] // splits):((i + 1) * f_pred.shape[0] // splits), :]
    kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    f_scores.append(np.exp(kl))
print('fake image IS:', np.mean(f_scores), np.std(f_scores))


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


# FID via fake pool3 and real pool3
mu1 = np.mean(f_act, axis=0)  # 2048
sigma1 = np.cov(f_act, rowvar=False)
mu2 = np.mean(r_act, axis=0)
sigma2 = np.cov(r_act, rowvar=False)  # (2048, 2048)
fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
print('FID:', fid_value)

