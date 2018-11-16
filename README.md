## MAML in pytorch

Performances are reported as best test accuracy. Converged test accuracy might be smaller than or similar to the reported performances.

### Loss curve



### Comparison to original MAML implementation for Omniglot

|      | 5-way 1-shot | 5-way 5-shot | 20-way 1-shot | 20-way 5-shot |
|:----:|:------------:|:------------:|:-------------:|:-------------:|
| MAML |     98.7%    |     99.9%    |     95.8%     |     98.9%     |
| Ours |              |              |               |       -       |

### Comparison to original MAML implementation for miniImageNet

|      | 5-way 1-shot | 5-way 5-shot |
|:----:|:------------:|:------------:|
| MAML |     48.7%    |     63.1%    |
| Ours |              |              |

### TODO
- Correctness check by benchmark experimental results (miniImageNet)
- Loss curve and score curve for example