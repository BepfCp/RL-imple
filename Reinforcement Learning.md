# 强化学习

## 导论



#### 符号表：

|      符号       | 含义                                                         |
| :-------------: | ------------------------------------------------------------ |
|  $\mathcal{S}$  | set of all nonterminal states                                |
| $\mathcal{S}^+$ | set of all states, including the terminal states             |
|       $T$       | final time step of an episode                                |
|    $\gamma$     | discount-rate parameter                                      |
|   $\pi(a|s)$    | probability of taking action $a$ in state $s$ under stochastic policy $\pi$ |
|  $\mathcal{A}$  | set of actions                                               |
|  $\mathcal{R}$  | set of all possible rewards, a finite subset of $\mathbb{R}$ |
|                 |                                                              |
|                 |                                                              |



## 有限马尔可夫决策过程

#### 智能体（agent)-环境交互

![](pic/agent_environment_interface.png)

在一个有限马尔可夫决策过程中，状态集、动作集以及奖赏元素都是有限的。

***马尔可夫性（Markov Property）***：在强化学习中，常被称作环境或MDP的动态性（Dynamics)
$$
p(s',r|s,a) \doteq \Pr\{S_t = s',R_t=r|S_{t-1}=s,A_{t-1}=a\}
$$
在利用强化学习解决实际问题时，如何选择动作和状态有时候更像是一门艺术，而非科学，更好的动作和状态集有时候对于问题的解决更有帮助；当然，如何设计奖赏函数更是一个极大的难题，一个基本的原则是：奖赏应当用来告诉智能体要做什么，而非怎么做。

#### `Return`($G_t$)

​	1）最简单的情形：
$$
G_t \doteq R_{t+1}+R_{t+2}+R_{t+3}+\dots+R_T
$$
​	2）折扣回报：
$$
G_t \doteq R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\dots = \sum_{k=0}^\infin \gamma^kR_{t+1+k} = R_{t+1}+\gamma G_{t+1}
$$

***统一***：将终止状态视作吸收状态，奖赏始终为0，即：

<img src="pic/absorbing_state.png" style="zoom:67%;" />

从而可以将回报统一写作：
$$
G_t \doteq \sum_{k=t+1}^T\gamma^{k-t-1}R_k
$$

#### 策略与价值函数

***策略***（policy）：从状态到执行该动作的概率的映射。

***价值函数***（value function）：

1）状态价值函数$v_\pi(s)$（state-value function for policy $\pi$）
$$
v_\pi(s) \doteq \mathbb{E}_\pi[G_t|S_t=s]=\mathbb{E}_\pi[\sum_{k=0}^\infin\gamma^kR_{t+k+1}|S_t=s]
$$
2）动作价值函数$q_\pi(s,a)$（action-value function for policy $\pi$）
$$
q_\pi(s,a) \doteq \mathbb{E}_\pi[G_t|S_t=s,A_t=a]=\mathbb{E}_\pi[\sum_{k=0}^\infin\gamma^kR_{t+k+1}|S_t=s,A_t=a]
$$

#### $v(s)$与$q(s,a)$的backup diagram

<center class="half">
    <img src="pic/v_backup.png" width="180"/>
    <img src="pic/q_backup.png" width="140"/>
</center>

*备注：上图中，每个空心圆圈表示一个状态；每个实心圆圈表示一个动作-状态对。*
$$
v_\pi(s) = \sum_a\pi(a|s)q_\pi(s,a)
$$

$$
\begin{equation}\begin{aligned}
q_\pi(s,a) &\doteq \mathbb{E}[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s,A_t=a]\\
&= \sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]
\end{aligned}\end{equation}
$$

#### 最优策略与最优价值函数

最优策略即能使智能体达到最大总期望奖赏的策略。最优策略拥有最优的状态价值函数和动作价值函数。
$$
v_*(s) \doteq \max_\pi v_\pi(s)
$$

$$
q_*(s,a) \doteq \max_\pi q_\pi(s,a)
$$

$$
q_*(s,a)=\mathbb{E}[R_{t+1}+\gamma v_*(S_{t+1})|S_t=s,A_t=a]
$$



#### `Bellman`方程：

$$
\begin{equation}\begin{aligned}
v_\pi(s) &\doteq \mathbb{E}_\pi[G_t|S_t=s]\\
&= \mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1})|S_t=s]\\
&= \mathbb{E}_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]\\
&= \sum_a \pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]\\
\end{aligned}\end{equation}
$$



#### `Bellman`最优方程：

$$
\begin{equation}\begin{aligned}
v_*(s) &= \max_{a\in \mathcal{A(s)}}q_*(s,a) \\
&= \max_a\mathbb{E}_{\pi_*}[G_t|S_t=s,A_t=a] \\
&= \max_a \mathbb{E}[R_{t+1}+\gamma v_*(S_{t+1})|S_t=s,A_t=a]\\
&= \max_a \sum_{s',r}p(s',r|s,a)[r+\gamma v_*(s')]
\end{aligned}\end{equation}
$$

$$
\begin{equation}\begin{aligned}q_*(s,a) &= \mathbb{E}[R_{t+1}+\gamma \max_{a'}q_*(S_{t+1},a')|S_t=s,A_t=a] \\&= \sum_{s',r}p(s',r|s,a)[r+\gamma \max_{a'}q_*(s',a')]\end{aligned}\end{equation}
$$

<img src="pic/backup_bellman.png" style="zoom: 80%;" />

直接求解贝尔曼最优方程的困难：

1）环境动态可能不尽知；

2）当前算力不足以应对指数级增长的状态和动作；

3）实际问题不满足马尔可夫性

## 动态规划

#### Policy Evaluation（prediction problem ）

> 对于给定的策略，计算状态价值函数。

利用`Bellman`方程，对于序列$v_0,v_1,v_2,\dots$，我们可以得到更新策略如下：
$$
\begin{equation}\begin{aligned}
v_{k+1}(s) &\doteq \mathbb{E}[R_{t+1}+\gamma v_k(S_{t+1})|S_t=s]\\
&= \sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]
\end{aligned}\end{equation}
$$
算法框图如下（in-place）:

<img src="pic/Iterative Policy Evaluation.png" style="zoom:70%;" />

#### Policy Improvement

> 通过在原有策略上改进，使新策略对价值函数更加贪心。

$$
\begin{equation}\begin{aligned}
\pi'(s) &\doteq \arg\max_a q_\pi(s,a)\\
&= \arg \max_a \mathbb{E}[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s,A_t=a] \\
&= \arg \max_a\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]
\end{aligned}\end{equation}
$$

#### Policy Iteration

<img src="pic/policy_iteration.png" style="zoom:70%;" />

#### Truncated Policy Iteration

1) Truncated Policy Evaluation

<img src="pic/truncated_policy_evaluation.png" style="zoom:95%;" />

2) Truncated Policy Iteration

![](pic/truncated_policy_iteration.png)

#### Value Iteration

> 实际上，价值更新（Value Iteration）就是把贝尔曼最优方程直接变成了更新规则；也可以将其视作截断策略迭代（Truncated Policy Iteration）的特殊情况：策略评估（Policy Evaluation）对每个状态只做一次交换（sweep）。

$$
\begin{equation}\begin{aligned}
v_{k+1}(s) &\doteq \max_a \mathbb{E}[R_{t+1}+\gamma v_k(S_{t+1})|S_t=s,A_t=a]\\
&= \max_a \sum_{s',r}p(s',r|s,a)[r+\gamma v_k(s')]
\end{aligned}\end{equation}
$$

<img src="pic/value_iteration.png" style="zoom:80%;" />

## 蒙特卡洛方法

