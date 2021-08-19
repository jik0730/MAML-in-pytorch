import argparse
import os
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
from time import time


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


def train_single_task(model, task_lr, loss_fn, dataloaders, args):
    """
    Train the model on a single few-shot task.
    We train the model with single or multiple gradient update.
    
    Args:
        model: (MetaLearner) a meta-learner to be adapted for a new task
        task_lr: (float) a task-specific learning rate
        loss_fn: a loss function
        dataloaders: (dict) a dict of DataLoader objects that fetches both of 
                     support set and query set
        args: (args) hyperparameters
    """
    # extract params
    num_train_updates = args.num_train_updates

    # support set and query set for a single few-shot task
    dl_sup = dataloaders['train']
    X_sup, Y_sup = dl_sup.__iter__().next()
    X_sup2, Y_sup2 = dl_sup.__iter__().next()

    X_sup, Y_sup = X_sup.to(args.device), Y_sup.to(args.device)

    adapted_state_dict = model.cloned_state_dict()  # NOTE what about just dict
    adapted_params = OrderedDict()
    for key, val in model.named_parameters():
        adapted_params[key] = val
        adapted_state_dict[key] = adapted_params[key]

    for _ in range(0, num_train_updates):
        Y_sup_hat = model(X_sup, adapted_state_dict)
        loss = loss_fn(Y_sup_hat, Y_sup)
        # print(len(list(adapted_params.values())))
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
                       args,
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
        args: (args) hyperparameters
        restore_file: (string) optional- name of file to restore from
                      (without its extension .pth.tar)
    TODO Validation classes
    """

    wandb.init(project='metadrop-pytorch', entity='joeljosephjin', config=vars(args))

    # params information
    num_classes = args.num_classes
    num_samples = args.num_samples
    num_query = args.num_query
    num_inner_tasks = args.num_inner_tasks
    task_lr = args.task_lr
    start_time = 0

    for episode in range(args.num_episodes):
        # Run inner loops to get adapted parameters (theta_t`)
        adapted_state_dicts = []
        dataloaders_list = []
        meta_loss = 0
        for n_task in range(num_inner_tasks):
            task = task_type(meta_train_classes, num_classes, num_samples, num_query)
            dataloaders = fetch_dataloaders(['train', 'test', 'meta'], task)
            a_dict = train_single_task(model, task_lr, loss_fn, dataloaders, args)

            dl_meta = dataloaders['meta']
            X_meta, Y_meta = dl_meta.__iter__().next()
            X_meta, Y_meta = X_meta.to(args.device), Y_meta.to(args.device)

            Y_meta_hat = model(X_meta, a_dict)
            loss_t = loss_fn(Y_meta_hat, Y_meta)
            meta_loss += loss_t
            
        meta_loss /= float(num_inner_tasks)

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        # Evaluate model on new task
        # Evaluate on train and test dataset given a number of tasks (args.num_steps)
        if (episode + 1) % args.save_summary_steps == 0:
            train_metrics = evaluate(model, loss_fn, meta_train_classes,
                                        task_lr, task_type, metrics, args,
                                        'train')
            test_metrics = evaluate(model, loss_fn, meta_test_classes,
                                    task_lr, task_type, metrics, args,
                                    'test')

            train_loss = train_metrics['loss']
            test_loss = test_metrics['loss']
            train_acc = train_metrics['accuracy']
            test_acc = test_metrics['accuracy']

            wandb.log({"episode":episode, "test_acc":test_acc, "train_acc":train_acc,"test_loss":test_loss,"train_loss":train_loss})
            print('episode: {:0.2f}, test_acc: {:0.2f}, train_acc: {:0.2f}, time: {:0.2f}, test_loss: {:0.2f}, train_loss: {:0.2f}'.format(episode, test_acc,train_acc,time()-start_time,test_loss,train_loss))
            start_time = time()

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()

    SEED = args.seed
    meta_lr = args.meta_lr
    num_episodes = args.num_episodes

    # Use GPU if available
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Set the random seed for reproducible experiments
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

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
    model = MetaLearner(args).to(args.device)

    # phi = model.phi
    
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    # fetch loss function and metrics
    loss_fn = nn.NLLLoss()
    model_metrics = metrics

    # Train the model
    train_and_evaluate(model, meta_train_classes, meta_test_classes, task_type,
                       meta_optimizer, loss_fn, model_metrics, args,
                       args.restore_file)
