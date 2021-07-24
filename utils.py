import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.manifold import TSNE
import seaborn as sns


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()


def calculate_predictions(model, dataloader, device):
    output_array = None
    label_array = None
    model.eval()
    with torch.no_grad():
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


def plot_learning_curve(history):
    NMI = history['NMI']
    ARI = history['ARI']
    ACC = history['ACC']
    total_loss_val = history['Total Loss val']

    total_loss = history['Total Loss']
    rec_loss = history['Reconstruction Loss']
    clus_loss = history['Clustering Loss']
    epochs = history['epochs']

    titles = ["NMI per Epoch", "ARI per Epoch", "Acc per Epoch"]

    ylables = ["NMI", "ARI", "Acc"]
    plot_data = [NMI, ARI, ACC]

    for ttl, ylbl, data in zip(titles, ylables, plot_data):
        plt.figure()
        plt.title(ttl)
        plt.plot(epochs, data, label='Train')
        plt.xlabel('Epoch')
        plt.ylabel(ylbl)

    plt.figure()
    plt.title("Total loss per Epoch")
    plt.plot(epochs, total_loss, label="Train")
    plt.plot(epochs, total_loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.legend()
    plt.show()

# TSNE plot
def run_model_and_get_tsne_losses(model, dataloader, device='cpu'):
    palette = sns.color_palette("bright", 10)
    output_distribution, labels, preds = calculate_predictions(model, dataloader, device)
    nmi = metrics.nmi(labels, preds)
    ari = metrics.ari(labels, preds)
    acc = metrics.acc(labels, preds)
    print('NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\n'.format(nmi, ari, acc))
    latent_numpy = None
    labels_numpy = None
    for data in dataloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        x, q, latent = model(inputs)

        if latent_numpy is None:
            latent_numpy = latent.cpu().detach().numpy()
            labels_numpy = labels.cpu().detach().numpy()
        else:
            latent_numpy = np.concatenate((latent_numpy, latent.cpu().detach().numpy()), 0)
            labels_numpy = np.concatenate((labels_numpy, labels.cpu().detach().numpy()), 0)

    tsne = TSNE()
    x_embedded = tsne.fit_transform(latent_numpy)
    ax = sns.scatterplot(x=x_embedded[:, 0], y=x_embedded[:, 1], hue=labels_numpy, legend='full', palette=palette)
    # ax.set_title(f'Epoch: {epoch}')
    plt.show()