# 笔记

## MCTS 

### PUCT

PUCT 用来选择 select 哪个 node 做 rollout

$\begin{aligned}U(s,a)&=Q(s,a)+c_{puct}P(s,a)\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}\end{aligned}$



### Select 

从root开始，找到 PUCT 最大的子节点，如果这个子节点是未访问过的节点，就立刻对它 rollout 然后回溯更新。
否则移动到这个子节点，重复上面的操作。

