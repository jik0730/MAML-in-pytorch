import unittest
import os
import copy

import torch
import torch.nn as nn
import utils
from src.model import MetaLearner


class test_meta_learner(unittest.TestCase):
    def setUp(self):
        # Configurations 3-way 3-shot with 3 query set
        model_dir = 'experiments/base_model'
        json_path = os.path.join(model_dir, 'params.json')
        assert os.path.isfile(
            json_path), "No json configuration file found at {}".format(
                json_path)
        params = utils.Params(json_path)

        params.in_channels = 3
        params.num_classes = 5
        params.dataset = 'ImageNet'
        self.model = MetaLearner(params)

        # Data setting
        N = 5
        self.X = torch.ones([N, params.in_channels, 84, 84])
        self.Y = torch.randint(params.num_classes, (N, ), dtype=torch.long)

        # Optim & loss setting
        self.optim = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.NLLLoss()

    def test_store_cur_params(self):
        # Store current parameters
        self.model.store_cur_params()

        # Update the model once with data
        Y_hat = self.model(self.X)
        loss = self.loss_fn(Y_hat, self.Y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Test stored_params deep copied
        for key, val in self.model.state_dict().items():
            self.assertTrue((val != self.model.stored_params[key]).any())

    # @unittest.skip("Adaptation process does not work..")
    def test_adapt_and_init_params(self):
        # Store current parameters
        self.model.store_cur_params()

        # Update the model once with data
        Y_hat = self.model(self.X)
        loss = self.loss_fn(Y_hat, self.Y)
        self.optim.zero_grad()
        # grads are in the order of model.parameters()
        grads = torch.autograd.grad(
            loss, self.model.parameters(), create_graph=True)
        # performs updates using calculated gradients
        # we manually compute adpated parameters since optimizer.step() operates in-place
        adapted_params = {
            key: val.clone()
            for key, val in self.model.state_dict().items()
        }
        for (key, val), grad in zip(self.model.named_parameters(), grads):
            adapted_params[key] = self.model.stored_params[key] - 1e-2 * grad

        # Check parameter not changed
        # self.model.check_params_not_changed()

        # Confirm that adapted_params are different from current params
        for key, val in self.model.named_parameters():
            self.assertTrue((val != adapted_params[key]).any())

        # Adapt adapted_params to the model
        # And confirm that adapted_params are the same to current params
        self.model.adapt_params(adapted_params)
        for key, val in adapted_params.items():
            self.assertTrue((val == self.model.state_dict()[key]).all())

        # Compute loss with adapted parameters
        # And optimize w.r.t. meta-parameters
        Y_hat = self.model(self.X)
        loss = self.loss_fn(Y_hat, self.Y)
        # Return to meta-parameters
        before_optim = copy.deepcopy(self.model.state_dict())
        self.model.init_params()
        meta_optim = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        # self.optim.zero_grad()
        meta_optim.zero_grad()
        loss.backward()
        # self.optim.step()
        meta_optim.step()

        # Check meta-parameters updated
        for key, val in self.model.named_parameters():
            self.assertTrue((val != before_optim[key]).any())

        # Check adapted-parameters still same

    @unittest.skip("For debugging purpose")
    def test_parameter_name(self):
        print(self.model.state_dict().keys())
        print(len(self.model.state_dict().keys()))
        print(self.model.meta_learner.state_dict().keys())
        print(len(self.model.meta_learner.state_dict().keys()))

        self.model.meta_learner.state_dict()['fc.bias'][0] = 0
        print(self.model.state_dict()['meta_learner.fc.bias'])
        print(self.model.meta_learner.state_dict()['fc.bias'])
        # for key in self.model.state_dict().keys():
        #     val1 = self.model.state_dict()[key]
        #     self.assertTrue(val is self.model.meta_learner.state_dict()[key])


if __name__ == '__main__':
    unittest.main()