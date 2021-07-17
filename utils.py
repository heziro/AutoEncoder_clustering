from const import *
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import albumentations as A
import torch
from scipy.optimize import linear_sum_assignment as linear_assignment

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()


def calculate_predictions(model, dataloader, device):
    output_array = None
    label_array = None
    model.eval()
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        _, outputs, _ = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            label_array = np.concatenate((label_array, labels.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
            label_array = labels.cpu().detach().numpy()

    preds = np.argmax(output_array.data, axis=1)
    # print(output_array.shape)
    return output_array, label_array, preds


def plot_image(plotter, img: np.ndarray):
    img_shape = img.shape
    if img_shape[-1] == 1:
        plotted_img = np.stack((img.squeeze(),)*3, axis=-1)
    else:
        plotted_img = img
    plotter.imshow(plotted_img)


class metrics:
    nmi = sklearn.metrics.normalized_mutual_info_score
    ari = sklearn.metrics.adjusted_rand_score

    @staticmethod
    def acc(labels_true, labels_pred):
        labels_true = labels_true.astype(np.int64)
        assert labels_pred.size == labels_true.size
        D = max(labels_pred.max(), labels_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(labels_pred.size):
            w[labels_pred[i], labels_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / labels_pred.size

def tsne_representation(clusters, labels, epoch, is_tensor):
    tsne  = TSNE()
    if is_tensor:
        labels_numpy = labels.detach().numpy()
        output_np = clusters.detach().numpy()
    else:
        labels_numpy = labels
        output_np = clusters
    x_embedded = tsne.fit_transform(output_np)
    ax = sns.scatterplot(x=x_embedded[:, 0], y=x_embedded[:, 1], hue=labels_numpy, legend='full', palette=palette)
    ax.set_title(f'Epoch: {epoch}')
    plt.show()


def augmentation(data,crop_size):
    data_list = []
    transform = A.Compose([A.RandomCrop(width=crop_size, height= crop_size),
                           A.HorizontalFlip(p=0.5),
                           A.RandomBrightnessContrast(p=0.2)])

    transform_2 = A.compose([A.augmentations.transforms.Blur(blur_limit=7, always_apply=False, p=0.5),
                             A.augmentations.transforms.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20,
                                                                  always_apply=False, p=0.5),
                             A.augmentations.transforms.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True,
                                                                    always_apply=False, p=0.5)])

    transform_3 = A.compose([A.augmentations.transforms.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8),
                                                               always_apply=False, p=0.5),A.HorizontalFlip(p=0.5)])

    for image in data:
        transformed = transform(image)['image']
        data_list.append(transformed)
        transformed_2 = transform_2(image)['image']
        data_list.append(transformed_2)
        transformed_3 = transform_3(image)['image']
        data_list.append(transformed_3)
    data_tensor = torch.Tensor(data_list)
    return data_tensor







