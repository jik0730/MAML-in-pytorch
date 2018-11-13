import unittest
import torch
import torch.nn as nn
from src.model import Net


class test_base_net(unittest.TestCase):
    def setUp(self):
        pass

    def test_input_dim_Omniglot(self):
        in_channels = 1
        num_classes = 5
        X = torch.ones([1, in_channels, 28, 28])
        em = Net(in_channels, num_classes, dataset='Omniglot')
        h_X = em.conv(X)
        self.assertTupleEqual(h_X.size(), (1, 64, 1, 1))
        f_X = em(X)
        self.assertTupleEqual(f_X.size(), (1, num_classes))

    def test_input_dim_miniImageNet(self):
        in_channels = 3
        num_classes = 5
        X = torch.ones([1, in_channels, 84, 84])
        em = Net(in_channels, num_classes, dataset='ImageNet')
        h_X = em.conv(X)
        self.assertTupleEqual(h_X.size(), (1, 64, 5, 5))
        f_X = em(X)
        self.assertTupleEqual(f_X.size(), (1, num_classes))

    def test_architecture(self):
        in_channels = 3
        num_classes = 5
        X = torch.ones([1, in_channels, 84, 84])
        Y = torch.ones([1, num_classes])
        em = Net(in_channels, num_classes, dataset='ImageNet')
        before_params = [p.clone() for p in em.parameters()]

        optimizer = torch.optim.Adam(em.parameters())
        loss_fn = nn.BCEWithLogitsLoss()

        f_X = em(X)
        loss = loss_fn(f_X, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after_params = [p.clone() for p in em.parameters()]

        for b_param, a_param in zip(before_params, after_params):
            # Make sure something changed.
            self.assertTrue((b_param != a_param).any())


if __name__ == '__main__':
    unittest.main()