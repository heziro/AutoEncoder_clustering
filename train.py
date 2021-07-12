import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torch.nn as nn
import models
import torch.optim as optim
from sklearn.cluster import KMeans


def load_data(batch_size=32, num_workers=0, small_trainset=False, n_samples=-1):

    train_set = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    if small_trainset:
        indices = np.where(train_set.targets == 0)[0][:int(n_samples / 2)]
        indices = np.concatenate((indices, torch.where(train_set.targets == 1)[0][:int(n_samples / 2)]))
        # Warp into Subsets and DataLoaders
        train_set = torch.utils.data.Subset(train_set, indices)
    else:
        if n_samples != -1:
            train_set = list(train_set)[:n_samples]

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def pretrained(model, train_loader, test_loader, rec_criterion, optimizer, batch_size, epochs, vis=False, device='cpu'):
    history = []
    model.train()
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            x, q, latent, p = model(inputs)
            # plt.imshow(inputs.detach().numpy()[0].reshape(28,28,1)); plt.show()
            # plt.imshow(x.detach().numpy()[0].reshape(28,28,1)); plt.show()
            loss = rec_criterion(x, inputs)
            loss.backward()
            optimizer.step()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))
        history.append((epoch, inputs, x), )

    if vis:
        model.eval()
        test_iter = iter(test_loader)
        images, labels = test_iter.next()
        outputs = model(images)

        plt.figure(figsize=(15, 15))
        n_examples = outputs[0].shape[0]
        for i in range(n_examples):
            img = images[i].reshape(28, 28, -1).detach().numpy()
            plt.subplot(n_examples, 2, 2 * i + 1)
            plt.title("Original img (pretrained)")
            plt.imshow(img)

            rec_img = outputs[0][i].reshape(28, 28, -1).detach().numpy()
            plt.subplot(n_examples, 2, 2 * i + 2)
            plt.title("Reconstruct img (pretrained)")
            plt.imshow(rec_img)
        plt.show()

    return model


def init_mu(model, train_loader, device):
    model.to(device)
    features = None
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        model.eval()
        # model.to(device)
        x, q, latent, p = model(inputs)
        if features is None:
            features = latent.cpu().detach().numpy()
        else:
            features = np.concatenate((features, latent.cpu().detach().numpy()), 0)
    km = KMeans(n_clusters=model.n_clusters, n_init=20)
    km.fit_predict(features)
    mu = torch.from_numpy(km.cluster_centers_)
    model.cluster_layer.set_mu(mu.to(device))

    model.train()
    return model


def train(model, train_loader, test_loader, rec_criterion, cluster_criterion, optimizer, batch_size, epochs, gamma,
          pretrained_epochs, vis=False, device='cpu'):
    model.to(device)

    model = pretrained(model, train_loader, test_loader, rec_criterion, optimizer, batch_size, pretrained_epochs,
                       vis=vis, device=device)
    # init mu with kmeans
    model = init_mu(model, train_loader, 'cpu')
    total_loss = 0
    all_points = []
    clusters = np.array([])
    model.train()
    history = []
    model.to(device)
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            x, q, latent, p = model(inputs)
            all_points.append(latent.cpu().detach().numpy())
            clusters = np.append(clusters, labels[0].cpu().detach().numpy())

            rec_loss = rec_criterion(x, inputs)
            clustering_loss = kl_loss(model.calc_p(q), q) / batch_size
            total_loss = rec_loss + gamma * clustering_loss
            total_loss.backward()
            optimizer.step()

        print('Epoch:{}, Total Loss:{:.4f}, Reconstruction Loss:{:.4f}, Clustering Loss:{:.4f}'.format(epoch + 1,
                                                   float(total_loss), float(rec_loss), float(clustering_loss*gamma)))
        history.append((epoch, inputs, x), )


        if vis and epoch % 50 == 0:
            plt.figure()
            all_points = np.array(all_points).reshape(-1, 2)
            scatter = plt.scatter(all_points[:, 0], all_points[:, 1], c=list(clusters.astype(np.int32)))
            plt.legend(handles=scatter.legend_elements()[0], labels=[0, 1])
            plt.show()

        all_points = []
        clusters = np.array([])

    model.eval()
    test_iter = iter(test_loader)
    images, labels = test_iter.next()
    outputs = model(images)

    plt.figure(figsize=(15, 15))
    n_examples = outputs[0].shape[0]
    for i in range(n_examples):
        img = images[i].reshape(28, 28, -1).detach().numpy()
        plt.subplot(n_examples, 2, 2 * i + 1)
        plt.title("Original img")
        plt.imshow(img)

        rec_img = outputs[0][i].reshape(28, 28, -1).detach().numpy()
        plt.subplot(n_examples, 2, 2 * i + 2)
        plt.title("Reconstruct img")
        plt.imshow(rec_img)
    plt.show()

    pass


def target(out_distr):
    tar_dist = out_distr ** 2 / torch.sum(out_distr, axis=0)
    tar_dist = torch.transpose(torch.transpose(tar_dist, 0, 1) / torch.sum(tar_dist, axis=1), 0, 1)
    return tar_dist


def kl_loss(p, q):
    return -1.0*nn.KLDivLoss(reduction='sum')(p, q)
    # return -1.0 * torch.sum(torch.sum(p*torch.log(p/q), dim=1), dim=0)


if __name__ == "__main__":
    model = models.ClusterAutoEncoder(input_shape=(32, 32, 1), n_clusters=10, dim=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    rec_criterion = nn.MSELoss()
    cluster_criterion = nn.MSELoss()
    batch_size = 64
    epochs = 20
    pretrained_epochs = 20
    gamma = 0.1
    small_trainset = False
    n_samples = -1
    vis = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device is: {device}')

    train_loader, test_loader = load_data(batch_size=batch_size, small_trainset=small_trainset, n_samples=n_samples)

    # get some random training images
    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()

    # show images
    # u.imshow(torchvision.utils.make_grid(images))

    train(model, train_loader, test_loader, rec_criterion, cluster_criterion, optimizer, batch_size, epochs, gamma,
          pretrained_epochs=pretrained_epochs, vis=vis, device=device)
