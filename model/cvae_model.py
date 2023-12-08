import torch
import torch.nn as nn


def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0, have_scene_flow=False):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels, have_scene_flow)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels, have_scene_flow)

    def forward(self, x, c=None, scene_flow=None):

        if x.dim() > 2:
            x = x.view(-1, 28*28)   # 在生成中常用的，先将图片view成向量

        means, log_var = self.encoder(x, c, scene_flow)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c, scene_flow)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)   # std指的是标准差
        eps = torch.randn_like(std)   # eps是从正态分布随机采样的

        return mu + eps * std

    def inference(self, z, c=None, scene_flow=None):

        recon_x = self.decoder(z, c, scene_flow)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, have_scene_flow):

        super().__init__()

        self.conditional = conditional
        self.have_scene_flow = have_scene_flow
        if self.conditional:
            layer_sizes[0] += num_labels
        if self.have_scene_flow:
            layer_sizes[0] += 3
        self.num_labels = num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None, scene_flow=None):

        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            x = torch.cat((x, c), dim=-1)
        
        if self.have_scene_flow:
            x = torch.cat((x, scene_flow), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, have_scene_flow):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        self.have_scene_flow = have_scene_flow
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size
            
        if self.have_scene_flow:
            input_size += 3
        self.num_labels = num_labels

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c, scene_flow=None):

        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            z = torch.cat((z, c), dim=-1)
        
        if self.have_scene_flow:
            z = torch.cat((z, scene_flow), dim=-1)

        x = self.MLP(z)

        return x
