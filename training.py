import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
import utils


def pretrained(model, train_loader, val_loader, rec_criterion, optimizer_pre, scheduler_pre, batch_size, epochs,
               vis=False, device='cpu', save=False, path=None):
    history = []

    save = True

    model.train()
    model.to(device)
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer_pre.zero_grad()
            x, q, latent = model(inputs)
            loss = rec_criterion(x, inputs)
            loss.backward()
            optimizer_pre.step()
        scheduler_pre.step()
        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))
        history.append((epoch, inputs, x), )

        if save:
            torch.save(model.state_dict(), path + "/model_cifar_" + str(epoch) + '.pth')
    if vis:
        model.eval()
        val_iter = iter(val_loader)
        images, labels = val_iter.next()
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
        x, q, latent = model(inputs)
        if features is None:
            features = latent.cpu().detach().numpy()
        else:
            features = np.concatenate((features, latent.cpu().detach().numpy()), 0)
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    km.fit_predict(features)
    mu = torch.from_numpy(km.cluster_centers_)
    model.clustering.set_mu(mu.to(device))

    model.train()
    return model


def train_supervise(model, train_loader, contrastive_criterion, con_gamma, optimizer, scheduler, device, epochs):
    model.to(device)
    model.clustering.weight = model.clustering.weight.to(device)
    for epoch in range(epochs):
        scheduler.step()
        model.train(True)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                x, q, latent = model(inputs)
                con_loss = contrastive_criterion(latent, model.clustering.weight[labels])
                total_loss = con_gamma * con_loss
                total_loss.backward()
                optimizer.step()
        print('Epoch:{}, contrastive Loss:{:.4f} '.format(epoch, total_loss))
    return model


def train(model, train_loader, val_loader, rec_criterion, cluster_criterion, optimizer, scheduler, batch_size,
          epochs, vis, device, pretrain, path, gamma):
    model.to(device)
    update_interval = 80
    tol = 1e-2
    finished = False

    total_loss = 0
    model.train()
    history = {'epochs': [],
               'loss': [],
               'NMI': [],
               'ARI': [],
               'ACC': [],
               'Total Loss': [],
               'Reconstruction Loss': [],
               'Clustering Loss': [],
               'Total Loss val': [],
               'Reconstruction Loss val': []
               }
    model.to(device)

    print("calculate prediction")
    output_distribution, labels, preds_prev = utils.calculate_predictions(model, train_loader, device)
    target_distribution = target(output_distribution)

    for epoch in range(epochs):
        scheduler.step()
        model.train(True)
        batch_num = 1
        for i, data in enumerate(train_loader, 0):
            if (batch_num - 1) % update_interval == 0 and not (batch_num == 1 and epoch == 0):
                output_distribution, labels, preds = utils.calculate_predictions(model, train_loader, device)
                target_distribution = target(output_distribution)
                nmi = utils.metrics.nmi(labels, preds)
                ari = utils.metrics.ari(labels, preds)
                acc = utils.metrics.acc(labels, preds)
                print('NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\n'.format(nmi, ari, acc))

                # check stop criterion
                delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
                preds_prev = np.copy(preds)
                if delta_label < tol:
                    print('Label divergence ' + str(delta_label) + '< tol ' + str(tol))
                    print('Reached tolerance threshold. Stopping training.')
                    finished = True
                    break

            tar_dist = target_distribution[((batch_num - 1) * batch_size):(batch_num * batch_size), :]
            tar_dist = torch.from_numpy(tar_dist).to(device)

            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                x, q, latent = model(inputs)

                rec_loss = rec_criterion(x, inputs)
                clustering_loss = gamma * cluster_criterion(torch.log(q), tar_dist) / batch_size
                total_loss = rec_loss + clustering_loss
                total_loss.backward()
                optimizer.step()
            batch_num = batch_num + 1

        print('Epoch:{}, Total Loss:{:.4f}, Reconstruction Loss:{:.4f},'
              ' Clustering Loss:{:.4f}'.format(epoch + 1, float(total_loss), float(rec_loss),
                                               float(clustering_loss * gamma)))

        history['epochs'].append(epoch + 1)
        history['NMI'].append(nmi)
        history['ARI'].append(ari)
        history['ACC'].append(acc)
        history['Total Loss'].append(total_loss)
        history['Reconstruction Loss'].append(rec_loss)
        history['Clustering Loss'].append(clustering_loss * gamma)

        # validation losses
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    x, q, latent = model(inputs)
                    rec_loss_val = rec_criterion(x, inputs)
                    # clustering_loss_val = gamma * cluster_criterion(torch.log(q), tar_dist) / batch_size
                    total_loss_val = rec_loss_val
                print('Epoch:{}, Validation: Total Loss:{:.4f}, '
                      'Reconstruction Loss:{:.4f}'.format(epoch + 1, float(total_loss_val), float(rec_loss_val)))

            history['Total Loss val'].append(total_loss_val)
            history['Reconstruction Loss val'].append(rec_loss_val)

        if finished:
            break

    return history


def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist
