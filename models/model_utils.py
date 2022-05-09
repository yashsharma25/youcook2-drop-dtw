import os
import torch
import torch.nn.functional as F
import numpy as np
from glob import glob

from paths import WEIGHTS_PATH


def cosine_sim(x, z):
    cos_sim_fn = torch.nn.CosineSimilarity(dim=1)
    return cos_sim_fn(x[..., None], z.T[None, ...])


def cos_dist(x, z):
    cos_sim_fn = torch.nn.CosineSimilarity(dim=1)
    return (1 - cos_sim_fn(x[..., None], z.T[None, ...])) / 2


def linear_sim(x, z):
    return x @ z.T


def l2_dist(x, z):
    dist_squared = (x ** 2).sum() + (z ** 2).sum() - 2 * linear_sim(x, z)
    return torch.clamp(dist_squared, min=0).sqrt()


def cos_loglikelihood(x, z, gamma=0.1, z_dim=1):
    cos_sim = cosine_sim(x, z)
    probs = F.softmax(cos_sim / gamma, dim=z_dim)
    return torch.log(probs)


def unique_softmax(sim, labels, gamma=1, dim=0):
    assert sim.shape[0] == labels.shape[0]
    labels = labels.detach().cpu().numpy()
    unique_labels, unique_index, unique_inverse_index = np.unique(
        labels, return_index=True, return_inverse=True)
    unique_sim = sim[unique_index]
    unique_softmax_sim = torch.nn.functional.softmax(unique_sim / gamma, dim=dim)
    softmax_sim = unique_softmax_sim[unique_inverse_index]
    return softmax_sim


def compute_normalization_parameters(dataset):
    mean_x, mean_z = torch.zeros(512), torch.zeros(1536)
    mean_x2, mean_z2 = torch.zeros(512), torch.zeros(1536)
    x_count, z_count = 0, 0
 
    #every s is a video-text tuple
    #every video contains multiple frames features
    #every text contains its embeddings

    counter = 0
    max_size = 0
    for s in dataset:
        #print(s['frame_features'].shape)
        if s['step_features'].shape[0] > max_size:
            max_size = s['step_features'].shape[0]
        # vid = video_text[0]   
        # text = video_text[1]

        # for frame_features, text_features in vid:

        # print("type of s[0] keys = ", type(s[0]))
        # print("Type of s = ", type(s))
        # print("Size of s =", len(s))
        # print("Type of first element in s = ", type(s[0]))
        # print("Length of first element in s = ", len(s[0]))
        # print("Shape of first element in s[0] = ", s[0][0].shape)

        # print("Type of second element in s = ", type(s[1]))
        # print("Length of second element in s = ", len(s[1]))
        # print("Shape of first element in s[1] = ", s[1][0].shape)

        mean_x += s['frame_features'].sum(0)
        mean_x2 += (s['frame_features'] ** 2).sum(0)
        x_count += s['frame_features'].shape[0]
                
        # print("s['step_features'] shape=", s['step_features'].shape)
        mean_z += s['step_features'].sum(0)
        mean_z2 += (s['step_features'] ** 2).sum(0)
        z_count += s['step_features'].shape[0]
        print("Normalising ", counter)
        counter+=1
        if counter == 100:
            break


    print("Max size = ", max_size)

    mean_x = mean_x / x_count
    mean_z = mean_z / z_count
    sigma_x = (mean_x2 / x_count - mean_x ** 2).sqrt()
    sigma_z = (mean_z2 / z_count - mean_z ** 2).sqrt()
    print("Completed normalisation")
    print("Returning ", mean_x.shape, sigma_x.shape, mean_z.shape, sigma_z.shape)
    return mean_x, sigma_x, mean_z, sigma_z


def load_last_checkpoint(name, model, device='cuda', strict=True,
                         remove_name_preffix=None, remove_name_postfix=None):
    weights_path = glob(os.path.join(WEIGHTS_PATH, name, f"weights-epoch=*.ckpt"))
    if len(weights_path) != 0:
        weights_path = weights_path[0]
    else:
        print(" NO checkpoint")
        return None
    state_dict = torch.load(weights_path, map_location=device)['state_dict']
    print(f"Loading checkpoint at {weights_path}")

    # adjust names in state dict
    new_keys = list(state_dict.keys())
    if remove_name_preffix:
        new_keys = [k[len(remove_name_preffix):] for k in new_keys]
    if remove_name_postfix:
        new_keys = [k[:-len(remove_name_preffix)] for k in new_keys]

    # load state dict with new keys
    new_state_dict = dict(zip(new_keys, state_dict.values()))
    model.load_state_dict(new_state_dict, strict=strict)
    return None
