import copy

import torch
import numpy as np
from collections import OrderedDict
from data.dataloader import fetch_dataloaders


def evaluate(model, loss_fn, meta_classes, task_lr, task_type, metrics, args,
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
    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for episode in range(args.num_steps):
        # Make a single task
        # Make dataloaders to load support set and query set
        task = task_type(meta_classes, args.num_classes, args.num_samples, args.num_query)
        dataloaders = fetch_dataloaders(['train', 'test'], task)
        dl_sup = dataloaders['train']
        dl_que = dataloaders['test']
        X_sup, Y_sup = dl_sup.__iter__().next()
        X_que, Y_que = dl_que.__iter__().next()

        X_sup, Y_sup = X_sup.to(args.device), Y_sup.to(args.device)
        X_que, Y_que = X_que.to(args.device), Y_que.to(args.device)

        # Direct optimization
        net_clone = copy.deepcopy(model)
        optim = torch.optim.SGD(net_clone.parameters(), lr=task_lr)
        for _ in range(args.num_eval_updates):
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

    return metrics_mean


if __name__ == '__main__':
    # TODO Evaluate trained model.
    pass