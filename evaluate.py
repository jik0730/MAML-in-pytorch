# Base code is from https://github.com/cs230-stanford/cs230-code-examples
import logging
import copy

import torch
import numpy as np
from torch.autograd import Variable
from src.data_loader import fetch_dataloaders


def evaluate(model, loss_fn, meta_classes, task_lr, task_type, metrics, params,
             split):
    """
    Evaluate the model on `num_steps` batches.
    
    Args:
        model: (MetaLearner) a meta-learner that is trained on MAML
        loss_fn: a loss function
        meta_classes: (list) a list of classes to be evaluated in meta-training or meta-testing
        task_lr: (float) a task-specific learning rate
        task_type: (subclass of FewShotTask) a type for generating tasks
        metrics: (dict) a dictionary of functions that compute a metric using 
                 the output and labels of each batch
        params: (Params) hyperparameters
        split: (string) 'train' if evaluate on 'meta-training' and 
                        'test' if evaluate on 'meta-testing' TODO 'meta-validating'
    """
    # params information
    SEED = params.SEED
    num_classes = params.num_classes
    num_samples = params.num_samples
    num_query = params.num_query
    num_steps = params.num_steps
    num_eval_updates = params.num_eval_updates

    # set model to evaluation mode
    # NOTE eval() is not needed since everytime task is varying and batchnorm
    # should compute statistics within the task.
    # model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for episode in range(num_steps):
        # Make a single task
        # Make dataloaders to load support set and query set
        task = task_type(meta_classes, num_classes, num_samples, num_query)
        dataloaders = fetch_dataloaders(['train', 'test'], task)
        dl_sup = dataloaders['train']
        dl_que = dataloaders['test']
        X_sup, Y_sup = dl_sup.__iter__().next()
        X_que, Y_que = dl_que.__iter__().next()

        # move to GPU if available
        if params.cuda:
            X_sup, Y_sup = X_sup.cuda(async=True), Y_sup.cuda(async=True)
            X_que, Y_que = X_que.cuda(async=True), Y_que.cuda(async=True)

        # # Adapt parameters by single gradient step
        # Y_sup_hat = model(X_sup)
        # loss = loss_fn(Y_sup_hat, Y_sup)
        # optimizer.zero_grad()
        # loss.backward()

        # # follows train_single_task
        # adapted_state_dict = model.cloned_state_dict()
        # for key, val in model.named_parameters():
        #     adapted_state_dict[key] = val - task_lr * val.grad

        # # compute preditions for query set
        # Y_que_hat = model(X_que, adapted_state_dict)

        # Direct optimization
        net_clone = copy.deepcopy(model)
        optim = torch.optim.SGD(net_clone.parameters(), lr=task_lr)
        for _ in range(num_eval_updates):
            Y_sup_hat = net_clone(X_sup)
            loss = loss_fn(Y_sup_hat, Y_sup)
            optim.zero_grad()
            loss.backward()
            optim.step()
        Y_que_hat = net_clone(X_que)
        loss = loss_fn(Y_que_hat, Y_que)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        Y_que_hat = Y_que_hat.data.cpu().numpy()
        Y_que = Y_que.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {
            metric: metrics[metric](Y_que_hat, Y_que)
            for metric in metrics
        }
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {
        metric: np.mean([x[metric] for x in summ])
        for metric in summ[0]
    }
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- [" + split.upper() + "] Eval metrics : " + metrics_string)

    return metrics_mean


if __name__ == '__main__':
    # TODO Evaluate trained model.
    pass