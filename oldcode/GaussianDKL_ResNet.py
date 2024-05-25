from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch import nn
import torch
import os
import torchvision.datasets as dset
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import gpytorch
import math
import tqdm
from kaggle_mod import MyDataset,read_split_data,TRANSFORMS

from torchvision.models import densenet

from torchvision.models import resnet50,ResNet50_Weights
from torchvision.models.resnet import ResNet, Bottleneck
from sklearn.model_selection import train_test_split
import sys
# Load pretrained weights for ResNet50
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Modify the last layer to have the desired number of output units
latent_layer = 1000
resnet.fc = nn.Linear(resnet.fc.in_features, latent_layer)


################ our DataLoaders
# r"C:\Users\grupo_gepar\Documents\lucho\Osteo\OsteoSAMTest"
# train_folder =r"C:\Users\grupo_gepar\Documents\lucho\Osteo\OsteoSAMTest\train"#r"C:\Users\grupo_gepar\Documents\lucho\Osteo\archivePrep\train"
# test_folder=r"C:\Users\grupo_gepar\Documents\lucho\Osteo\OsteoSAMTest\test"#r"C:\Users\grupo_gepar\Documents\lucho\Osteo\archivePrep\val"

# train_data = datasets.ImageFolder(train_folder, transform=transforms.Compose([transforms.ToTensor()]))
# #datasets.ImageFolder(r'C:\Users\grupo_gepar\Documents\lucho\Osteo\Osteoarthritis_Assignment_dataset\train', transform=transforms.Compose([transforms.ToTensor()]))
# test_data = datasets.ImageFolder(test_folder, transform=transforms.Compose([transforms.ToTensor()]))
# #datasets.ImageFolder(r'C:\Users\grupo_gepar\Documents\lucho\Osteo\Osteoarthritis_Assignment_dataset\Valid', transform=transforms.Compose([transforms.ToTensor()]))
# print(f"Data tensor Dimension:",train_data[0][0].shape)
# #Convert to DataLoader
# train_loader = DataLoader(train_data, shuffle=True, batch_size=1)
# test_loader = DataLoader(test_data, shuffle=True, batch_size=1)

# num_classes=len(train_loader.dataset.classes)

# print((train_loader.dataset.class_to_idx))


######## CODE FROM KAGGLE GUY

root_path=r"C:\Users\grupo_gepar\Documents\lucho\Osteo\Osteoarthritis_Assignment_Merged"

train_image_path, train_image_label, val_image_path, val_image_label, class_indices = read_split_data(root_path,False)

train_dataset = MyDataset(train_image_path, train_image_label, TRANSFORMS['train'])
valid_dataset = MyDataset(val_image_path, val_image_label, TRANSFORMS['valid'])
assert set(train_image_path).intersection(set(val_image_path)) == set()
num_classes= len(class_indices)
batch_size=1 # we only support 1

system_name = sys.platform
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) if 'linux' in system_name else 0
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
                            num_workers=nw, collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, pin_memory=True,
                            num_workers=nw, collate_fn=valid_dataset.collate_fn)
test_loader = valid_loader
#################################





base_model = ResNet(Bottleneck, [3, 4, 6, 3])

"""
#Creating the DenseNet Model
class DenseNetFeatureExtractor(densenet.DenseNet):
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out

feature_extractor = DenseNetFeatureExtractor(block_config=(6, 6, 6), num_classes=num_classes)
num_features = feature_extractor.classifier.in_features
print(num_features)
"""


class ResNetFeatureExtractor(ResNet):
    def __init__(self, num_classes):
        super(ResNetFeatureExtractor, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.fc = nn.Linear(self.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    


feature_extractor = resnet #ResNetFeatureExtractor(num_classes=1024)#num_classes)

# Freeze the parameters of all the layers except for the last one
freeze = False
if freeze:
    print("freezing resnet layers")
    for param in feature_extractor.parameters():
        param.requires_grad = True

# Set the last layer to be trainable
num_ftrs = feature_extractor.fc.in_features
out_features = 1000
feature_extractor.fc = nn.Linear(num_ftrs, out_features)

num_features = feature_extractor.fc.out_features
print(num_features)

#Creating the Gaussian Process Layer
class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


#Creating the DKL Model
class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res
    
model = DKLModel(feature_extractor, num_dim=num_features)
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, num_classes=num_classes)


# If you run this example without CUDA, I hope you like waiting!
if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()
    print(torch.cuda.is_available())

# Set up the optimizer
n_epochs = 1000
lr = 0.001
samples_likelihood=16
optimizer = SGD([
    {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
    {'params': model.gp_layer.hyperparameters(), 'lr': lr },
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.01)
mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))
import numpy as np

#Training and Testing
def train(epoch):
    model.train()
    likelihood.train()
    correct = 0
    losses = []
    minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    with gpytorch.settings.num_likelihood_samples(samples_likelihood):
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = -mll(output, target)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())

            # ACC
            output = likelihood(model(data))  # This gives us 16 samples from the predictive distribution
            pred = output.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            correct += pred.eq(target.view_as(pred)).cpu().sum()
            losses.append(loss.item())
    acc= 100*correct/len(train_loader.dataset)
    loss=np.mean(losses)
    return acc,loss
import collections

# We'll train the model for 10 epochs
def test():
    model.eval()
    likelihood.eval()

    correct = 0
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(samples_likelihood):
        preds_history=[]
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = likelihood(model(data))  # This gives us 16 samples from the predictive distribution
            pred = output.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            correct += pred.eq(target.view_as(pred)).cpu().sum()
            preds_history.append(pred.item())

    acc= correct/len(test_loader.dataset)
    str_acc=f"Test set: Accuracy: {correct}/{len(test_loader.dataset)} {100. * acc}%"
    print(str_acc)
    print('pred set')#,set(preds_history))
    unipreds=collections.Counter(preds_history)
    print(unipreds)
    print({x:100*y/float(len(test_loader.dataset)) for x,y in unipreds.items()})
    return 100*acc
acc_best = -1
for epoch in range(1, n_epochs + 1):
    with gpytorch.settings.use_toeplitz(False):
        train_acc,train_loss=train(epoch)
        test_acc = test()
    print(f"epoch {epoch}")
    print(f"train loss : {train_loss}")
    print(f"train acc : {train_acc}")
    print(f"test acc {test_acc}")
    scheduler.step()
    state_dict = model.state_dict()
    likelihood_state_dict = likelihood.state_dict()
    if test_acc > acc_best:
        acc_best = test_acc
        torch.save({'model': state_dict, 
                    'likelihood': likelihood_state_dict,
                    'best-epoch': epoch,
                    "testacc":acc_best,
                    "testacc":acc_best,
                    "trainacc":train_acc,
                    "trainloss":train_loss}, "_checkpoint.dat")