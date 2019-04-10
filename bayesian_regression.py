import os
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn

import matplotlib.pyplot as plt; plt.ion()
import pyro
from pyro import distributions as dist
from pyro.distributions import Normal, Uniform, Delta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions.util import logsumexp
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
import pyro.optim as optim
import pyro.poutine as poutine

# for CI testing
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('0.3.0')
pyro.enable_validation(True)
pyro.set_rng_seed(1)
pyro.enable_validation(True)


#####################
# Load Data         #
#####################
DATA_URL = "https://d2fefpcigoriu7.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = data[data["cont_africa"] == 1]
non_african_nations = data[data["cont_africa"] == 0]
ax[0].scatter(non_african_nations["rugged"],
    np.log(non_african_nations["rgdppc_2000"]))
ax[0].set(xlabel="Terrain Ruggedness Index",
      ylabel="log GDP (2000)",
      title="Non African Nations")
ax[1].scatter(african_nations["rugged"],
    np.log(african_nations["rgdppc_2000"]))
ax[1].set(xlabel="Terrain Ruggedness Index",
      ylabel="log GDP (2000)",
      title="African Nations")

data = torch.tensor(df.values, dtype=torch.float)
x_data, y_data = data[:, :-1], data[:, -1]


##################################################################
# Standard PyTorch Linear Regression Module, direct optimization #
##################################################################
class RegressionModel(nn.Module):
    """ Linear regression with one interaction term
            y ~ x1 + x1 + x1*x2
    """
    def __init__(self, p):
        # p = number of features
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)
        self.factor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return self.linear(x) + (self.factor * x[:,0] * x[:,1]).unsqueeze(1)

# instantiate model
p = 2  # number of features
regression_model = RegressionModel(p)
for name, param in regression_model.named_parameters():
    print(name, param.data.numpy())

# create loss
loss_fn = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(regression_model.parameters(), lr=0.05)
num_iterations = 2000

for j in range(num_iterations):
    # run the model forward on the data
    y_pred = regression_model(x_data).squeeze(-1)
    # calculate the mse loss
    loss = loss_fn(y_pred, y_data)
    # initialize gradients to zero
    optim.zero_grad()
    # backpropagate
    loss.backward()
    # take a gradient step
    optim.step()
    if (j + 1) % 50 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))

# Inspect learned parameters
print("\nLearned parameters:")
for name, param in regression_model.named_parameters():
    print(name, param.data.numpy())


#############################################
# "Lifting" a pytorch module into pyro      #
#############################################

# define a unit normal prior
loc   = torch.zeros(1, 1)
scale = torch.ones(1, 1)
prior = Normal(loc, scale)

# overload the parameters in the regression module with samples from the prior
lifted_module = pyro.random_module("regression_model", regression_model, prior)

# sample a nn from the prior
for i in range(3):
    sampled_reg_model = lifted_module()
    print("\nsample %d parameters"%i)
    for name, param in sampled_reg_model.named_parameters():
        print(name, param.data.numpy())


#####################################################################
# quick aside on PYRO random samples (I think this is similar to    #
# how Edward does it as well)                                       #
#####################################################################
prior = dist.Normal(torch.zeros(2, 3, 4, 5), torch.ones(2, 3, 4, 5))
print("batch vs event shapes:", prior.batch_shape, prior.event_shape)
samp = prior.sample()
print(samp.shape)
lls  = prior.log_prob(samp)
print(lls.shape)

print("make final two dimensions (4, 5) are now event shape. first two (2, 3) are batch shape")
prior = prior.to_event(2)
print("batch vs event shapes:", prior.batch_shape, prior.event_shape)
samp = prior.sample()
print(samp.shape)
lls = prior.log_prob(samp)
print(lls.shape)


#############################################
# Bayesian Linear Regression w/ Pyro        #
#############################################

def model(x_data, y_data):
    # weight and bias priors
    w_prior = Normal(torch.zeros(1, 2), torch.ones(1, 2)).to_event(1)
    b_prior = Normal(torch.tensor([[8.]]), torch.tensor([[1000.]])).to_event(1)
    f_prior = Normal(0., 1.)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior, 'factor': f_prior}
    scale = pyro.sample("sigma", Uniform(0., 10.))
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a nn (which also samples w and b)
    lifted_reg_model = lifted_module()
    with pyro.plate("map", len(x_data)):
        # run the nn forward on data
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        # condition on the observed data
        pyro.sample("obs",
                    Normal(prediction_mean, scale),
                    obs=y_data)
        return prediction_mean

#
# Set up "Guide", i.e. the variational family
#
from pyro.contrib.autoguide import \
    AutoDiagonalNormal, \
    AutoMultivariateNormal, \
    AutoIAFNormal, \
    AutoDelta

guide = AutoDiagonalNormal(model)
#guide = AutoMultivariateNormal(model)
#guide = AutoDelta(model)

#
# optimize variational parameters
#
optim = Adam({"lr": 0.02})
svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=1000)
pyro.clear_param_store()
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(x_data, y_data)
    if j % 100 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))

#
# Compare Optimization to Bayesian Posterior
#
def get_marginal(traces, sites):
    return EmpiricalMarginal(traces, sites).\
        _get_samples_and_weights()[0].detach().cpu().numpy()

posterior = svi.run(x_data, y_data)

weight = get_marginal(posterior, ['module$$$linear.weight']).squeeze(1).squeeze(1)
factor = get_marginal(posterior, ['module$$$factor'])
gamma_within_africa = weight[:, 1] + factor.squeeze(1)
gamma_outside_africa = weight[:, 1]
fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
sns.distplot(gamma_within_africa, kde_kws={"label": "African nations"},)
sns.distplot(gamma_outside_africa, kde_kws={"label": "Non-African nations"})
fig.suptitle("Density of Slope : log(GDP) vs. Terrain Ruggedness", fontsize=16)

lweight = regression_model.linear.weight.detach().squeeze()
lfactor = regression_model.factor.item()
lgamma_within = lweight[1].item() + lfactor
lgamma_outside = lweight[1].item()
ax.scatter(lgamma_within, 0, s=100, marker='x', label='lst sq within' )
ax.scatter(lgamma_outside, 0, s=100, marker='x', label='lst sq outside')
ax.legend()


#################################################################
# Same Model, Different (more "pyro") implementation            #
#################################################################
from torch.distributions import constraints
train = torch.tensor(df.values, dtype=torch.float)
is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]
sites = ["a", "bA", "bR", "bAR", "sigma"]

def model(is_cont_africa, ruggedness, log_gdp):
    a = pyro.sample("a", dist.Normal(8., 1000.))
    b_a = pyro.sample("bA", dist.Normal(0., 1.))
    b_r = pyro.sample("bR", dist.Normal(0., 1.))
    b_ar = pyro.sample("bAR", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
    mean = a + b_a * is_cont_africa + \
        b_r * ruggedness + b_ar * is_cont_africa * ruggedness
    with pyro.iarange("data", len(ruggedness)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)

#
# Diagonal Normal Guide, explicitly programmed as a Pyro Model/Function
#
def guide(is_cont_africa, ruggedness, log_gdp):
    a_loc = pyro.param('a_loc', torch.tensor(0.))
    a_scale = pyro.param('a_scale', torch.tensor(1.),
                         constraint=constraints.positive)
    sigma_loc = pyro.param('sigma_loc', torch.tensor(1.),
                             constraint=constraints.positive)
    weights_loc = pyro.param('weights_loc', torch.randn(3))
    weights_scale = pyro.param('weights_scale', torch.ones(3),
                               constraint=constraints.positive)
    a = pyro.sample("a", dist.Normal(a_loc, a_scale))
    b_a = pyro.sample("bA", dist.Normal(weights_loc[0], weights_scale[0]))
    b_r = pyro.sample("bR", dist.Normal(weights_loc[1], weights_scale[1]))
    b_ar = pyro.sample("bAR", dist.Normal(weights_loc[2], weights_scale[2]))
    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
    mean = a + b_a * is_cont_africa + \
        b_r * ruggedness + b_ar * is_cont_africa * ruggedness


#
# Aside --- what is an effect handler?  what is a trace??
#
# poutine is pyro's library of effect handlers
poutine

# traces wrap stochastic function
traced_model = poutine.trace(model)
mtrace = traced_model.get_trace(is_cont_africa, ruggedness, log_gdp)
mtrace.log_prob_sum()
print(mtrace.nodes['bA'])
print(mtrace.nodes['bAR'])
print(mtrace.nodes['obs'])

# trace of guide!?
traced_guide = poutine.trace(guide)
traced_guide(is_cont_africa, ruggedness, log_gdp)
gtrace = traced_guide.get_trace(is_cont_africa, ruggedness, log_gdp)
gtrace.log_prob_sum()
gtrace.nodes['a_loc']
gtrace.nodes['bA']


#
# Diagonal SVI
#
svi = SVI(model,
          guide,
          Adam({"lr": .005}),
          loss=Trace_ELBO(),
          num_samples=1000)
pyro.clear_param_store()
num_iters = 8000 if not smoke_test else 2
for i in range(num_iters):
    elbo = svi.step(is_cont_africa, ruggedness, log_gdp)
    if i % 500 == 0:
        print("Elbo loss: {}".format(elbo))

svi_diagnorm_posterior = svi.run(log_gdp, is_cont_africa, ruggedness)


#
# Run w/ NUTS
#
nuts_kernel = NUTS(model, adapt_step_size=True)
hmc_posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200) \
    .run(is_cont_africa, ruggedness, log_gdp)

#for site, values in summary(hmc_posterior, sites).items():
#    print("Site: {}".format(site))
#    print(values, "\n")

#
# Run w/ Multivariate Normal
#
mvn_guide = AutoMultivariateNormal(model)
svi = SVI(model,
          mvn_guide,
          Adam({"lr": .005}),
          loss=Trace_ELBO(),
          num_samples=1000)
pyro.clear_param_store()
for i in range(num_iters):
    elbo = svi.step(is_cont_africa, ruggedness, log_gdp)
    if i % 500 == 0:
        print("Elbo loss: {}".format(elbo))

svi_mvn_posterior = svi.run(log_gdp, is_cont_africa, ruggedness)


#####################################
# Compare All Marginals             #
#####################################

svi_diagnorm_empirical = \
    EmpiricalMarginal(svi_diagnorm_posterior, sites=sites) \
        ._get_samples_and_weights()[0] \
        .detach().cpu().numpy()

hmc_empirical = \
    EmpiricalMarginal(hmc_posterior, sites=sites) \
        ._get_samples_and_weights()[0].numpy()

svi_mvn_empirical = \
    EmpiricalMarginal(svi_mvn_posterior, sites=sites) \
        ._get_samples_and_weights()[0] \
        .detach().cpu().numpy()

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle("Marginal Posterior density - Regression Coefficients", fontsize=16)
for i, ax in enumerate(axs.reshape(-1)):
    sns.distplot(svi_diagnorm_empirical[:, i], ax=ax, label="SVI (DiagNormal)", hist=False)
    sns.distplot(hmc_empirical[:, i], ax=ax, label="HMC", hist=False)
    sns.distplot(svi_mvn_empirical[:,i], ax=ax, label='MVN', hist=False)
    ax.set_title(sites[i])
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')


##################################################################
# Other Pyro Stuff!
#  - plate notation + subsampling
#  - amortization 
#  - enumeration of discrete states (mixtures, sequence models)
#     - see deep markov model: https://pyro.ai/examples/dmm.html
#  - jit (torchscript?)
#  - LDS models
##################################################################

