import numpy as np
import matplotlib.pyplot as plt
import sklearn
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