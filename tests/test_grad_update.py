import unittest
import os
import copy
from datetime import datetime

import torch
import torch.nn as nn
import utils
from src.model import MetaLearner


class test_grad_update(unittest.TestCase):
    def setUp(self):
        # Configurations 3-way 3-shot with 3 query set
        model_dir = 'experiments/base_model'
        json_path = os.path.join(model_dir, 'params.json')
        assert os.path.isfile(
            json_path), "No json configuration file found at {}".format(
                json_path)
        self.params = utils.Params(json_path)

        self.params.in_channels = 3
        self.params.num_classes = 5
        self.params.dataset = 'ImageNet'
        self.model = MetaLearner(self.params)

        # Data setting
        N = 5
        self.X = torch.ones([N, self.params.in_channels, 84, 84])
        self.Y = torch.randint(
            self.params.num_classes, (N, ), dtype=torch.long)

        # Optim & loss setting
        self.loss_fn = nn.NLLLoss()

    @unittest.skip("too complicated test; may not be a correct approach")
    def test_inner_and_meta_update(self):
        # Store current parameters
        self.model.store_cur_params()

        # Update the model once with data
        meta_optim = torch.optim.SGD(self.model.stored_params, lr=0.1)
        Y_hat = self.model(self.X)
        loss = self.loss_fn(Y_hat, self.Y)
        meta_optim.zero_grad()

        # grads are in the order of model.parameters()
        grads = torch.autograd.grad(
            loss, self.model.parameters(), create_graph=True)

        # performs updates using calculated gradients
        # we manually compute adpated parameters since optimizer.step() operates in-place
        adapted_params = {
            key: val
            for key, val in self.model.stored_params.items()
        }
        for (key, val), grad in zip(self.model.named_parameters(), grads):
            adapted_params[key] = self.model.stored_params[key] - 1e-2 * grad

        # Check parameter not changed
        self.model.check_params_not_changed()

        # Confirm that adapted_params are different from current params
        for key, val in self.model.named_parameters():
            self.assertTrue((val != adapted_params[key]).any())

        #################################
        ### META-UPDATE
        #################################
        # # clone a model (for debugging)
        # model_init = copy.deepcopy(self.model)

        # # load adapted_params to model
        # self.model.adapt_params(adapted_params)
        # a = copy.deepcopy(self.model.state_dict()['meta_learner.fc.weight'])

        # # clone a model (for debugging)
        # model_adap = copy.deepcopy(self.model)

        # # check current loaded params differ from the original params
        # for key, val in self.model.named_parameters():
        #     self.assertTrue((val != self.model.stored_params[key]).any())

        # # compute loss using adapted_params
        # start = datetime.now()
        # Y_hat = self.model(self.X)
        # loss = self.loss_fn(Y_hat, self.Y)
        # interval = (datetime.now() - start).total_seconds()

        # # load again original params
        # self.model.init_params()
        # b = copy.deepcopy(self.model.state_dict()['meta_learner.fc.weight'])

        # # check current loaded params differ from the adapted_params
        # for key, val in self.model.named_parameters():
        #     self.assertTrue((val != adapted_params[key]).any())

        # # update original params using the loss computed
        # self.model.store_cur_params()

        # start = datetime.now()
        # meta_optim = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        # meta_optim.zero_grad()
        # loss.backward()
        # meta_optim.step()
        # print('meta {}'.format(interval +
        #                        (datetime.now() - start).total_seconds()))

        # # check original params updated
        # c = copy.deepcopy(self.model.state_dict()['meta_learner.fc.weight'])
        # print(a)
        # print(b)
        # print(c)
        # for key, val in self.model.named_parameters():
        #     # print(key)
        #     # print(val)
        #     # print(self.model.stored_params[key])
        #     self.assertTrue((val != self.model.stored_params[key]).any())

        # # TODO does this procedure really compute hessian??
        # # update model_init
        # start = datetime.now()
        # Y_hat_init = model_init(self.X)
        # loss_init = self.loss_fn(Y_hat_init, self.Y)
        # init_optim = torch.optim.SGD(model_init.parameters(), lr=1e-3)
        # init_optim.zero_grad()
        # loss_init.backward()
        # init_optim.step()
        # print('init {}'.format((datetime.now() - start).total_seconds()))
        # for key, val in self.model.named_parameters():
        #     self.assertTrue((val != model_init.state_dict()[key]).any())

        # # update model_adap
        # start = datetime.now()
        # Y_hat_adap = model_adap(self.X)
        # loss_adap = self.loss_fn(Y_hat_adap, self.Y)
        # adap_optim = torch.optim.SGD(model_adap.parameters(), lr=1e-3)
        # adap_optim.zero_grad()
        # loss_adap.backward()
        # adap_optim.step()
        # print('adap {}'.format((datetime.now() - start).total_seconds()))
        # for key, val in self.model.named_parameters():
        #     self.assertTrue((val != model_adap.state_dict()[key]).any())

    @unittest.skip("loading state_dict might break computational graph")
    def test_train_maml_not_working_1(self):
        start = datetime.now()
        Y_hat = self.model(self.X)
        loss = self.loss_fn(Y_hat, self.Y)

        meta_optim = torch.optim.SGD(self.model.parameters(), lr=0.1)
        meta_optim.zero_grad()
        loss.backward(create_graph=True)
        print((datetime.now() - start).total_seconds())
        adapted_state_dict = self.model.cloned_state_dict()
        for key, val in self.model.named_parameters():
            adapted_state_dict[key] = val - 1e-2 * val.grad

        # define another meta-learner with adpated_state_dict
        # NOTE this approach is not working!!
        task_learner = MetaLearner(self.params)
        task_learner.load_state_dict(adapted_state_dict)

        # compute loss with task_learner and optim by original params
        start = datetime.now()
        Y_hat_task = task_learner(self.X)
        loss_task = self.loss_fn(Y_hat_task, self.Y)
        meta_optim.zero_grad()
        for key, val in self.model.named_parameters():
            print(key)
            print(val.grad)
        loss_task.backward()
        for key, val in self.model.named_parameters():
            print(key)
            print(val.grad)
        print((datetime.now() - start).total_seconds())

    def test_train_maml(self):
        """
        This might be a correct approach.
        """
        start = datetime.now()
        Y_hat = self.model(self.X)
        loss = self.loss_fn(Y_hat, self.Y)

        meta_optim = torch.optim.SGD(self.model.parameters(), lr=0.1)
        meta_optim.zero_grad()
        loss.backward(create_graph=True)
        print('\n 1st gradient computation takes {}'.format(
            (datetime.now() - start).total_seconds()))
        adapted_state_dict = self.model.cloned_state_dict()
        for key, val in self.model.named_parameters():
            adapted_state_dict[key] = val - 1e-2 * val.grad

        # compute loss using adapted params and optim by original params
        start = datetime.now()
        Y_hat_task = self.model(self.X, adapted_state_dict)
        loss_task = self.loss_fn(Y_hat_task, self.Y)
        meta_optim.zero_grad()
        loss_task.backward()
        # for key, val in self.model.named_parameters():
        #     print(key)
        #     print(val.grad)
        print('2nd gradient computation takes {}'.format(
            (datetime.now() - start).total_seconds()))

    @unittest.skip("okay get a concept")
    def test_simple_maml_case(self):
        """
        What we found:
            If create_graph=True, w.grad.requires_grad is True.
            If create_graph=False, w.grad.requires_grad is False.
        """
        print('\n')
        x = torch.tensor(1.)
        y = torch.tensor(1.)
        w = torch.tensor(2., requires_grad=True)

        # If change w
        w_c = torch.tensor(2., requires_grad=True)

        loss = (y - w * x)**2
        optim = torch.optim.SGD([w], lr=0.1)
        optim.zero_grad()
        print('[1st] w-before {}'.format(w.grad))  # 0
        loss.backward(create_graph=True)
        print('[1st] w-after {}'.format(w.grad))  # 2
        print(w.grad.requires_grad)

        w_ = w - 0.1 * w.grad
        w_c_ = w_c - 0.1 * w.grad

        loss_ = (y - w_ * x)**2
        optim.zero_grad()
        print('[2nd] w-before {}'.format(w.grad))  # 0
        loss_.backward(retain_graph=True)
        print('[2nd] w-after {}'.format(w.grad))  # 1.28

        loss_c = (y - w_c_ * x)**2
        optim.zero_grad()
        print('[2nd] w-before {}'.format(w_c.grad))  # 0
        loss_c.backward()
        print('[2nd] w-after {}'.format(w_c.grad))  # 1.6


if __name__ == '__main__':
    unittest.main()