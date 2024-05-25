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

from torchvision.models import densenet
from torchvision.models import resnet50


#Create DataLoaders
train_data = datasets.ImageFolder(r'C:\Users\grupo_gepar\Documents\lucho\Osteo\Processing\train', transform=transforms.Compose([transforms.ToTensor()]))
test_data = datasets.ImageFolder(r'C:\Users\grupo_gepar\Documents\lucho\Osteo\Processing\val', transform=transforms.Compose([transforms.ToTensor()]))
print(f"Data tensor Dimension:",train_data[0][0].shape)

#Convert to DataLoader
train_loader = DataLoader(train_data, shuffle=True, batch_size=1)
test_loader = DataLoader(test_data, shuffle=True, batch_size=1)
num_classes=len(train_loader.dataset.classes)

print((train_loader.dataset.class_to_idx))


#Creating the DenseNet Model
# class DenseNetFeatureExtractor(densenet.DenseNet):
#     def forward(self, x):
#         features = self.features(x)
#         out = F.relu(features, inplace=True)
#         out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
#         return out

# feature_extractor = DenseNetFeatureExtractor(block_config=(6, 6, 6), num_classes=num_classes)
# num_features = feature_extractor.classifier.in_features
# print(num_features)

#Creating the ResNet50 Model
class ResNet50FeatureExtractor(resnet50.ResNet50):
    def forward(self, x):
        features = self.fc(self.conv1(x))
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out

feature_extractor = ResNet50FeatureExtractor(pretrained=True)
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

# Set up the optimizer
n_epochs = 100
lr = 0.001
optimizer = SGD([
    {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.01)
mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))


#Training and Testing
def train(epoch):
    model.train()
    likelihood.train()

    minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    with gpytorch.settings.num_likelihood_samples(8):
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = -mll(output, target)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())

# We'll train the model for 10 epochs
def test():
    model.eval()
    likelihood.eval()

    correct = 0
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = likelihood(model(data))  # This gives us 16 samples from the predictive distribution
            pred = output.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    print('Test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
    ))

for epoch in range(1, n_epochs + 1):
    with gpytorch.settings.use_toeplitz(False):
        train(epoch)
        test()
    scheduler.step()
    state_dict = model.state_dict()
    likelihood_state_dict = likelihood.state_dict()
    torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, '_checkpoint.dat')