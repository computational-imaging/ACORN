import torch
from torch import nn
import numpy as np
import math
from functools import partial


class Sine(nn.Module):
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.w0 * input)


class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None, w0=30):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(w0=w0), partial(sine_init, w0=w0), first_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            nn.Linear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, out_features), nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords):
        output = self.net(coords)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True, normalize=False,
                 input_dim=3, gaussian_pe=False, gaussian_variance=38):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = nn.Parameter(gaussian_variance * torch.randn(num_encoding_functions, input_dim),
                                                 requires_grad=False)
        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0,
                    self.num_encoding_functions - 1,
                    self.num_encoding_functions)
            else:
                self.frequency_bands = torch.linspace(
                    2.0 ** 0.0,
                    2.0 ** (self.num_encoding_functions - 1),
                    self.num_encoding_functions)

            if normalize:
                self.normalization = torch.tensor(1/self.frequency_bands)

    def forward(self, tensor) -> torch.Tensor:
        r"""Apply positional encoding to the input.

        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).

        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:
                    if self.normalization is not None:
                        encoding.append(self.normalization[idx]*func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor w0
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class ImplicitAdaptivePatchNet(nn.Module):
    def __init__(self, in_features=3, out_features=1, feature_grid_size=(8, 8, 8),
                 hidden_features=256, num_hidden_layers=3, patch_size=8,
                 code_dim=8, use_pe=True, num_encoding_functions=6, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feature_grid_size = feature_grid_size
        self.patch_size = patch_size
        self.use_pe = use_pe

        if self.use_pe:
            self.positional_encoding = PositionalEncoding(num_encoding_functions=num_encoding_functions)
            in_features = 2*in_features*num_encoding_functions + in_features

        self.coord2features_net = FCBlock(in_features=in_features, out_features=np.prod(feature_grid_size),
                                          num_hidden_layers=num_hidden_layers, hidden_features=hidden_features,
                                          outermost_linear=True, nonlinearity='relu')

        self.features2sample_net = FCBlock(in_features=self.feature_grid_size[0], out_features=out_features,
                                           num_hidden_layers=1, hidden_features=64,
                                           outermost_linear=True, nonlinearity='relu')
        print(self)

    def forward(self, model_input):

        # Enables us to compute gradients w.r.t. coordinates
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        fine_coords = model_input['fine_rel_coords'].clone().detach().requires_grad_(True)

        if self.use_pe:
            coords = self.positional_encoding(coords)

        features = self.coord2features_net(coords)

        # features is size (Batch Size, Blocks, prod(feature_grid_size))
        # but currently interpolate bilinear only supports one batch dimension,
        # therefore, for now assume that Batch Size == 1
        assert features.shape[0] == 1, 'Code currently only supports Batch Size == 1'

        n_channels, dx, dy = self.feature_grid_size
        features = features.squeeze(0)
        b_size = features.shape[0]

        features_in = features.squeeze().reshape(b_size, n_channels, dx, dy)
        sample_coords_out = fine_coords[0, ...].reshape(1, -1, 2)
        sample_coords = sample_coords_out.reshape(b_size, self.patch_size[0], self.patch_size[1], 2)

        y = sample_coords[..., :1]
        x = sample_coords[..., 1:]
        sample_coords = torch.cat([y, x], dim=-1)

        features_out = torch.nn.functional.grid_sample(features_in, sample_coords,
                                                       mode='bilinear',
                                                       padding_mode='border',
                                                       align_corners=True).reshape(b_size, n_channels, np.prod(self.patch_size))

        # permute from (Blocks, feature_grid_size[0], patch_size**2)->(Blocks, patch_size**2, feature_grid_size[0])
        # so the network maps features to function output
        features_out = features_out.permute(0, 2, 1)

        # for all spatial feature vectors, extract function value
        patch_out = self.features2sample_net(features_out)

        # squeeze out last dimension and restore batch dimension
        patch_out = patch_out.unsqueeze(0)

        return {'model_in': {'sample_coords_out': sample_coords_out, 'model_in_coarse': coords},
                'model_out': {'output': patch_out, 'codes': None}}


class ImplicitAdaptiveOctantNet(nn.Module):
    def __init__(self, in_features=4, out_features=1, feature_grid_size=(4, 16, 16, 16),
                 hidden_features=256, num_hidden_layers=3, octant_size=8,
                 code_dim=8, use_pe=True, num_encoding_functions=6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feature_grid_size = feature_grid_size
        self.octant_size = octant_size
        self.use_pe = use_pe

        if self.use_pe:
            self.positional_encoding = PositionalEncoding(num_encoding_functions=num_encoding_functions)
            in_features = 2*in_features*num_encoding_functions + in_features

        self.coord2features_net = FCBlock(in_features=in_features, out_features=np.prod(feature_grid_size),
                                          num_hidden_layers=num_hidden_layers, hidden_features=hidden_features,
                                          outermost_linear=True, nonlinearity='relu')

        self.features2sample_net = FCBlock(in_features=feature_grid_size[0], out_features=out_features,
                                           num_hidden_layers=1, hidden_features=64,
                                           outermost_linear=True, nonlinearity='relu')

    def forward(self, model_input, oversample=1.0):

        # Enables us to compute gradients w.r.t. coordinates
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        fine_coords = model_input['fine_rel_coords'].clone().detach().requires_grad_(True)

        if self.use_pe:
            coords = self.positional_encoding(coords)

        features = self.coord2features_net(coords)

        # features is size (Batch Size, Blocks, prod(feature_grid_size))
        # but currently interpolate bilinear only supports one batch dimension,
        # therefore, for now assume that Batch Size == 1
        assert features.shape[0] == 1, 'Code currently only supports Batch Size == 1'

        n_channels, dx, dy, dz = self.feature_grid_size
        features = features.squeeze(0)
        b_size = features.shape[0]

        features_in = features.squeeze().reshape(b_size, n_channels, dx, dy, dz)
        sample_coords_out = fine_coords[0, ...].reshape(1, -1, 3)
        sample_coords = sample_coords_out.reshape(b_size, self.octant_size, self.octant_size, self.octant_size, 3)
        features_out = torch.nn.functional.grid_sample(features_in, sample_coords,
                                                       mode='bilinear',
                                                       padding_mode='border',
                                                       align_corners=True).reshape(b_size, n_channels, self.octant_size**3)

        # permute from (Blocks, feature_grid_size[0], patch_size**2)->(Blocks, patch_size**2, feature_grid_size[0])
        # so the network maps features to function output
        features_out = features_out.permute(0, 2, 1)

        # for all spatial feature vectors, extract function value
        patch_out = self.features2sample_net(features_out)

        # squeeze out last dimension and restore batch dimension
        patch_out = patch_out.unsqueeze(0)

        return {'model_in': {'sample_coords_out': sample_coords_out, 'model_in_coarse': coords},
                'model_out': {'output': patch_out, 'codes': None}}


class ImplicitNet(nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, sidelength, out_features=1, in_features=2,
                 mode='pe', hidden_features=256, num_hidden_layers=3, w0=30, **kwargs):

        super().__init__()
        self.mode = mode

        if self.mode == 'pe':
            nyquist_rate = 1 / (2 * (2 * 1/np.max(sidelength)))
            num_encoding_functions = int(math.floor(math.log(nyquist_rate, 2)))

            nonlinearity = 'relu'
            self.positional_encoding = PositionalEncoding(num_encoding_functions=num_encoding_functions)
            in_features = 2*in_features*num_encoding_functions + in_features

        elif self.mode == 'siren':
            nonlinearity = 'sine'
        else:
            raise NotImplementedError(f'mode=={self.mode} not implemented')

        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=nonlinearity, w0=w0)
        print(self)

    def forward(self, model_input):

        coords = model_input['fine_abs_coords'][..., :2]

        if self.mode == 'pe':
            coords = self.positional_encoding(coords)

        output = self.net(coords)
        return {'model_in': {'coords': coords}, 'model_out': {'output': output}}
