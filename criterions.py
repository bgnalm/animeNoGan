import torch

class EdgeLoss(torch.nn.Module):
    def __init__(self, alpha=0.1):
        super(EdgeLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = torch.nn.L1Loss()
        self.edge_l1_loss = torch.nn.L1Loss()
        self.laplacian = torch.Tensor([
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ],
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ],
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ],
        ])

        self.laplacian = self.laplacian.view(1,3,3,3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.laplacian = self.laplacian.to(device)

    def forward(self, inputs, targets):
        input_laplacian = torch.nn.functional.conv2d(inputs, self.laplacian)
        target_laplacian = torch.nn.functional.conv2d(targets, self.laplacian)
        return self.l1_loss(targets, inputs) + (self.alpha * self.edge_l1_loss(target_laplacian, input_laplacian))


"""
Copied example
reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD
"""


class VAELoss(torch.nn.Module):

    def __init__(self):
        self.reconstruction_function = torch.nn.BCELoss()

    def forward(self, network_out, x):
        recon_x, mu, logvar = network_out # make it generic, so the forward is always input and target
        BCE = self.reconstruction_function(recon_x, x)

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return BCE + KLD


if __name__ == '__main__':
    # test
    input_img = torch.ones(4, 3, 224, 224)
    output = torch.zeros(4, 3, 224, 224)
    edge_loss = EdgeLoss()
    edge_loss(input_img, output)