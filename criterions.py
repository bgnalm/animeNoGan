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

    def forward(self, inputs, targets):
        input_laplacian = torch.nn.functional.conv2d(inputs, self.laplacian)
        target_laplacian = torch.nn.functional.conv2d(targets, self.laplacian)
        return self.l1_loss(targets, inputs) + (self.alpha * self.edge_l1_loss(target_laplacian, input_laplacian))

if __name__ == '__main__':
    # test
    input_img = torch.ones(4, 3, 224, 224)
    output = torch.zeros(4, 3, 224, 224)
    edge_loss = EdgeLoss()
    edge_loss(input_img, output)