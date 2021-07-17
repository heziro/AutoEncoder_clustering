from sklearn.manifold import TSNE
import seaborn as sns


in_channels = 1
semisupervised_update_interval = 16
gamma_labelled_loss = 0.01
w_img = 28
h_img = 28
lr = 0.001
weight_decay = 0.0
sched_gamma = 0.1
sched_step = 200
batch_size = 256
epochs = 200
pretrained_epochs = 300
gamma = 0.1
small_trainset = False
n_samples = -1
dataset = "MNIST"
#ataset = "CIFAR-10"
tsne  = TSNE()
palette = sns.color_palette("bright", 10)
batch_tsne = 1
