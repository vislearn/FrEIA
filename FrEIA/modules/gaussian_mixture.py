from . import InvertibleModule

import torch
import torch.nn.functional as F


class GaussianMixtureModel(InvertibleModule):
    '''An invertible Gaussian mixture model. The weights, means, covariance
    parameterization and component index must be supplied as conditional inputs
    to the module and can come from an external feed-forward network, which may
    be trained by backpropagating through the GMM. Weights should first be
    normalized via GaussianMixtureModel.normalize_weights(w) and component
    indices can be sampled via GaussianMixtureModel.pick_mixture_component(w).
    If component indices are specified, the model reduces to that Gaussian
    mixture component and maps between data x and standard normal latent
    variable z. Components can also be chosen consistently at random, by
    supplying an integer random seed instead of indices. If a None value is
    supplied instead of indices, the model maps between K data points x and K
    latent codes z simultaneously, where K is the number of mixture components.
    Mathematical derivations are found in the technical report "Training Mixture
    Density Networks with full covariance matrices" on arXiv.'''

    def __init__(self, dims_in, dims_c):
        super().__init__(dims_in, dims_c)

        self.x_dims = dims_in[0][0]
        # Prepare masks for filling the (triangular) Cholesky factors of the precision matrices
        self.mask_upper = (torch.triu(torch.ones(self.x_dims, self.x_dims), diagonal=1) == 1)
        self.mask_diagonal = torch.eye(self.x_dims, self.x_dims).bool()


    @staticmethod
    def pick_mixture_component(w, seed=None):
        '''Randomly choose mixture component indices with probability given by
        the component weights w. Works on batches of component weights.

        w:      Weights of the mixture components, must be positive and sum to one
        seed:   Optional RNG seed for consistent decisions'''

        w_thresholds = torch.cumsum(w, dim=1)
        # Prepare local random number generator
        rng = torch.Generator(device=w.device)
        if isinstance(seed, int):
            rng = rng.manual_seed(seed)
        else:
            rng.seed()
        # Draw one uniform random number per batch row and compare against thresholds
        u = torch.rand(w.shape[0], 1, device=w.device, generator=rng)
        indices = torch.sum(u > w_thresholds, dim=1).int()
        # Return mixture component indices
        return indices


    @staticmethod
    def normalize_weights(w):
        '''Apply softmax to ensure component weights are positive and sum to
        one. Works on batches of component weights.

        w:  Unnormalized weights for Gaussian mixture components, must be of
            size [batch_size, n_components]'''

        return F.softmax(w - w.max(), dim=-1)


    @staticmethod
    def nll_loss(w, z, log_jacobian):
        '''Negative log-likelihood loss for training a Mixture Density Network.

        w:              Mixture component weights, must be positive and sum to
                        one. Tensor must be of size [batch_size, n_components].
        z:              Latent codes for all mixture components. Tensor must be
                        of size [batch, n_components, n_dims].
        log_jacobian:   Jacobian log-determinants for each precision matrix.
                        Tensor size must be [batch_size, n_components].'''

        return -((-0.5 * (z**2).sum(dim=-1) + log_jacobian).exp() * w).sum(dim=-1).log()


    @staticmethod
    def nll_upper_bound(w, z, log_jacobian):
        '''Numerically more stable upper bound of the negative log-likelihood
        loss for training a Mixture Density Network.

        w:              Mixture component weights, must be positive and sum to
                        one. Tensor must be of size [batch_size, n_components].
        z:              Latent codes for all mixture components. Tensor must be
                        of size [batch, n_components, n_dims].
        log_jacobian:   Jacobian log-determinants for each precision matrix.
                        Tensor size must be [batch_size, n_components].'''

        return -(w.log() - 0.5 * (z**2).sum(dim=-1) + log_jacobian).sum(dim=-1)


    def forward(self, x, c, rev=False, jac=True):
        '''Map between data distribution and standard normal latent distribution
        of mixture components or entire mixture, in an invertible way.

        x:  Data during forward pass or latent codes during backward pass. Size
            must be [batch_size, n_dims] if component indices i are specified
            and should be [batch_size, n_components, n_dims] if not.

        The conditional input c must be a list [w, mu, U, i] of parameters for
        the Gaussian mixture model with the following properties:

        w:  Weights of the mixture components, must be positive and sum to one
            and have size [batch_size, n_components].
        mu: Means of the mixture components, must have size [batch_size,
            n_components, n_dims].
        U:  Entries for the (upper triangular) Cholesky factors for the
            precision matrices of the mixture components. These are needed to
            parameterize the covariance of the mixture components and must have
            size [batch_size, n_components, n_dims * (n_dims + 1) / 2].
        i:  Tensor of component indices (size [batch_size]), or a single integer
            to be used as random number generator seed for component selection,
            or None to indicate that all mixture components are modelled.'''
        assert len(x) == 1, f"GaussianMixtureModel got {len(x)} inputs, but " \
                            f"only one is allowed."
        x = x[0]

        # Get GMM parameters
        w, mu, U_entries, i = c
        batch_size, n_components = w.shape

        # Construct upper triangular Cholesky factors U of all precision matrices
        U = torch.zeros(batch_size, n_components, self.x_dims, self.x_dims, device=x.device)
        # Fill everything above the diagonal as is
        U[self.mask_upper.expand(batch_size,n_components,-1,-1)] = U_entries[:,:,self.x_dims:].reshape(-1)
        # Diagonal entries must be positive
        U[self.mask_diagonal.expand(batch_size,n_components,-1,-1)] = U_entries[:,:,:self.x_dims].exp().reshape(-1)

        # Indices of chosen mixture components, if provided
        if i is None:
            fixed_components = False
        else:
            fixed_components = True
            if not isinstance(i, torch.Tensor):
                i = self.pick_mixture_component(w, seed=i)

        if jac: 
            # Compute Jacobian log-determinants
            # Note: we avoid a log operation by taking diagonal entries directly from U_entries, where they are in log space
            if fixed_components:
                # Keep Jacobian log-determinants for chosen components only
                j = torch.stack([U_entries[b, i[b], :self.x_dims].sum(dim=-1) for b in range(batch_size)])
            else:
                # Keep Jacobian log-determinants for all components simultaneously
                j = U_entries[:, :, :self.x_dims].sum(dim=-1)

            if rev:
                j *= -1
        else:
            j = None

        # Actual forward and inverse pass
        if not rev:
            if fixed_components:
                # Return latent codes of x according to chosen component distributions only
                return [torch.stack([torch.matmul(U[b,i[b],:,:], x[b,:] - mu[b,i[b],:]) for b in range(batch_size)])], j
            else:
                # Return latent codes of x according to all component distributions simultaneously
                if len(x.shape) < 3:
                    x = x[:,None,:]
                return [torch.matmul(U, (x - mu)[...,None])[...,0]], j
        else:
            if fixed_components:
                # Transform latent samples to samples from chosen mixture distributions
                return [torch.stack([mu[b,i[b],:] + torch.matmul(torch.inverse(U[b,i[b],:,:]), x[b,:]) for b in range(batch_size)])], j
            else:
                # Transform latent samples to samples from all mixture distributions simultaneously
                return [torch.matmul(torch.inverse(U), x[...,None])[...,0] + mu], j

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims
