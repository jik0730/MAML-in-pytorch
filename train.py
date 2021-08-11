# Base code is from https://github.com/cs230-stanford/cs230-code-examples
import argparse
import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from model import MetaLearner
from model import Net
from model import metrics
from data.dataloader import split_omniglot_characters
from data.dataloader import load_imagenet_images
from data.dataloader import OmniglotTask
from data.dataloader import ImageNetTask
from data.dataloader import fetch_dataloaders
from evaluate import evaluate

import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',default='data/Omniglot',help="Directory containing the dataset")
parser.add_argument('--model_dir',default='experiments/base_model',help="Directory containing params.json")
parser.add_argument('--restore_file',default=None,help="Optional, init model weight file")  # 'best' or 'train'
parser.add_argument('--seed',default=1)
parser.add_argument('--dataset',default="Omniglot")
parser.add_argument('--meta_lr',default=1e-3, type=float)
parser.add_argument('--task_lr',default=1e-1, type=float)
parser.add_argument('--num_episodes',default=10000, type=int)
parser.add_argument('--num_classes',default=5, type=int)
parser.add_argument('--num_samples',default=1, type=int)
parser.add_argument('--num_query',default=10, type=int)
parser.add_argument('--num_steps',default=100, type=int)
parser.add_argument('--num_inner_tasks',default=8, type=int)
parser.add_argument('--num_train_updates',default=1, type=int)
parser.add_argument('--num_eval_updates',default=3, type=int)
parser.add_argument('--save_summary_steps',default=100, type=int)
parser.add_argument('--num_workers',default=1, type=int)


def train_single_task(model, task_lr, loss_fn, dataloaders, params):
    """
    Train the model on a single few-shot task.
    We train the model with single or multiple gradient update.
    
    Args:
        model: (MetaLearner) a meta-learner to be adapted for a new task
        task_lr: (float) a task-specific learning rate
        loss_fn: a loss function
        dataloaders: (dict) a dict of DataLoader objects that fetches both of 
                     support set and query set
        params: (Params) hyperparameters
    """
    # extract params
    num_train_updates = params.num_train_updates

    # set model to training mode
    model.train()

    # support set and query set for a single few-shot task
    dl_sup = dataloaders['train']
    X_sup, Y_sup = dl_sup.__iter__().next()
    X_sup2, Y_sup2 = dl_sup.__iter__().next()

    # move to GPU if available
    if args.cuda:
        X_sup, Y_sup = X_sup.cuda(), Y_sup.cuda()

    # compute model output and loss
    Y_sup_hat = model(X_sup)
    loss = loss_fn(Y_sup_hat, Y_sup)

    # clear previous gradients, compute gradients of all variables wrt loss
    def zero_grad(params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    # NOTE if we want approx-MAML, change create_graph=True to False
    # optimizer.zero_grad()
    # loss.backward(create_graph=True)
    zero_grad(model.parameters())
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # performs updates using calculated gradients
    # we manually compute adpated parameters since optimizer.step() operates in-place
    adapted_state_dict = model.cloned_state_dict()  # NOTE what about just dict
    adapted_params = OrderedDict()
    for (key, val), grad in zip(model.named_parameters(), grads):
        adapted_params[key] = val - task_lr * grad
        adapted_state_dict[key] = adapted_params[key]

    for _ in range(1, num_train_updates):
        Y_sup_hat = model(X_sup, adapted_state_dict)
        loss = loss_fn(Y_sup_hat, Y_sup)
        zero_grad(adapted_params.values())
        # optimizer.zero_grad()
        # loss.backward(create_graph=True)
        grads = torch.autograd.grad(
            loss, adapted_params.values(), create_graph=True)
        for (key, val), grad in zip(adapted_params.items(), grads):
            adapted_params[key] = val - task_lr * grad
            adapted_state_dict[key] = adapted_params[key]

    return adapted_state_dict


def train_and_evaluate(model,
                       meta_train_classes,
                       meta_test_classes,
                       task_type,
                       meta_optimizer,
                       loss_fn,
                       metrics,
                       params,
                       model_dir,
                       restore_file=None):
    """
    Train the model and evaluate every `save_summary_steps`.

    Args:
        model: (MetaLearner) a meta-learner for MAML algorithm
        meta_train_classes: (list) the classes for meta-training
        meta_train_classes: (list) the classes for meta-testing
        task_type: (subclass of FewShotTask) a type for generating tasks
        meta_optimizer: (torch.optim) an meta-optimizer for MetaLearner
        loss_fn: a loss function
        metrics: (dict) a dictionary of functions that compute a metric using 
                 the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from
                      (without its extension .pth.tar)
    TODO Validation classes
    """

    wandb.init(project='metadrop-pytorch', entity='joeljosephjin', config=vars(params))

    # params information
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query
    num_inner_tasks = params.num_inner_tasks
    task_lr = params.task_lr
    meta_lr = params.meta_lr

    # TODO validation accuracy
    best_test_acc = 0.0

    # For plotting to see summerized training procedure
    plot_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    with tqdm(total=params.num_episodes) as t:
        for episode in range(params.num_episodes):
            # Run one episode
            logging.info("Episode {}/{}".format(episode + 1,
                                                params.num_episodes))

            # Run inner loops to get adapted parameters (theta_t`)
            adapted_state_dicts = []
            dataloaders_list = []
            for n_task in range(num_inner_tasks):
                task = task_type(meta_train_classes, num_classes, num_samples,
                                 num_query)
                dataloaders = fetch_dataloaders(['train', 'test', 'meta'],
                                                task)
                # Perform a gradient descent to meta-learner on the task
                a_dict = train_single_task(model, task_lr, loss_fn,
                                           dataloaders, params)
                # Store adapted parameters
                # Store dataloaders for meta-update and evaluation
                adapted_state_dicts.append(a_dict)
                dataloaders_list.append(dataloaders)

            # Update the parameters of meta-learner
            # Compute losses with adapted parameters along with corresponding tasks
            # Updated the parameters of meta-learner using sum of the losses
            meta_loss = 0
            for n_task in range(num_inner_tasks):
                dataloaders = dataloaders_list[n_task]
                dl_meta = dataloaders['meta']
                X_meta, Y_meta = dl_meta.__iter__().next()
                if args.cuda:
                    X_meta, Y_meta = X_meta.cuda(), Y_meta.cuda(
                        )

                a_dict = adapted_state_dicts[n_task]
                Y_meta_hat = model(X_meta, a_dict)
                loss_t = loss_fn(Y_meta_hat, Y_meta)
                meta_loss += loss_t
            meta_loss /= float(num_inner_tasks)
            # print(meta_loss.item())

            # Meta-update using meta_optimizer
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()

            # Evaluate model on new task
            # Evaluate on train and test dataset given a number of tasks (params.num_steps)
            if (episode + 1) % params.save_summary_steps == 0:
                train_metrics = evaluate(model, loss_fn, meta_train_classes,
                                         task_lr, task_type, metrics, params,
                                         'train')
                test_metrics = evaluate(model, loss_fn, meta_test_classes,
                                        task_lr, task_type, metrics, params,
                                        'test')

                train_loss = train_metrics['loss']
                test_loss = test_metrics['loss']
                train_acc = train_metrics['accuracy']
                test_acc = test_metrics['accuracy']

                wandb.log({"episode":episode, "test_acc":test_acc, "train_acc":train_acc,"test_loss":test_loss,"train_loss":train_loss})
                print('episode: {:0.2f}, test_acc: {:0.2f}, train_acc: {:0.2f}, test_loss: {:0.2f}, train_loss: {:0.2f}'.format(episode, test_acc,train_acc,test_loss,train_loss))


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()

    SEED = args.seed
    meta_lr = args.meta_lr
    num_episodes = args.num_episodes

    # Use GPU if available
    args.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(SEED)
    if args.cuda: torch.cuda.manual_seed(SEED)

    # NOTE These params are only applicable to pre-specified model architecture.
    # Split meta-training and meta-testing characters
    if 'Omniglot' in args.data_dir and args.dataset == 'Omniglot':
        args.in_channels = 1
        meta_train_classes, meta_test_classes = split_omniglot_characters(
            args.data_dir, SEED)
        task_type = OmniglotTask
    elif ('miniImageNet' in args.data_dir or
          'tieredImageNet' in args.data_dir) and args.dataset == 'ImageNet':
        args.in_channels = 3
        meta_train_classes, meta_test_classes = load_imagenet_images(
            args.data_dir)
        task_type = ImageNetTask
    else:
        raise ValueError("I don't know your dataset")

    # Define the model and optimizer
    if args.cuda:
        model = MetaLearner(args).cuda()
    else:
        model = MetaLearner(args)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    # fetch loss function and metrics
    loss_fn = nn.NLLLoss()
    model_metrics = metrics

    # Train the model
    train_and_evaluate(model, meta_train_classes, meta_test_classes, task_type,
                       meta_optimizer, loss_fn, model_metrics, args,
                       args.model_dir, args.restore_file)
