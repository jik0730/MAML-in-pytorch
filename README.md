# MAML in pytorch

Performances are reported as best test accuracy. 
Converged test accuracy might be smaller than or similar to the reported performances.

## Comparison to original MAML implementation for Omniglot

|      | 5-way 1-shot | 5-way 5-shot | 20-way 1-shot | 20-way 5-shot |
|:----:|:------------:|:------------:|:-------------:|:-------------:|
| MAML |     98.7%    |     99.9%    |     95.8%     |     98.9%     |
| Ours |     99.4%    |     99.9%    |     92.8%     |       -       |

## Comparison to original MAML implementation for miniImageNet

|      | 5-way 1-shot | 5-way 5-shot |
|:----:|:------------:|:------------:|
| MAML |     48.7%    |     63.1%    |
| Ours |     48.4%    |     64.8%    |

## TODO
- Investigate under-performance of 20-way 1-shot omniglot experiments (This is similar behavior to that of https://github.com/katerakelly/pytorch-maml)