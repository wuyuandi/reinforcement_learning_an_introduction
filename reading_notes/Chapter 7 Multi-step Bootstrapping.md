---
typora-copy-images-to: \images
---

# Chapter 7 Multi-step Bootstrapping

在前两章我们介绍了蒙特卡洛算法和TD算法，其实这两种算法都不是最优的。现在我们介绍一种多步TD算法，这种算法可以随意的变换到TD(0)算法或蒙特卡洛算法。多步TD算法有着比蒙特卡洛算法和TD(0)算法更好的表现。

多步TD算法的另一个好处是它可以使我们不必为单个时间片产生的麻烦而烦恼。在许多应用中，人们希望能够快速的更新行为来获得事物改变的原因，所以更新程序的时间片足够长，那么它就越可能包含有明显作用的状态改变的这一过程，那么更新程序的效果就越好。多步TD算法可以使我们不必为单个时间片产生的麻烦而烦恼。

与以前一样，我们首先考虑预测问题，然后在考虑控制问题，最后推广到行为价值的预测和控制问题。

## n-step TD Prediction

蒙特卡洛算法是接受直到终止状态的整个奖励序列。单步TD算法接受本次的奖励，对于后续的奖励，使用$V(S_{t+1})$来代替。在蒙特卡洛算法和单步TD算法之间，我们可以改变TD算法接受的时间片的长度，使时间片的长度大于1而小于整个奖励序列的长度。例如，两步TD算法，就是接受奖励$R_{t+1} $和$R_{t+2}$，剩下的奖励用$V(S_{t+2})$来代替。同样，我们也可以有三步TD算法、四步TD算法等。

![1508036684810](images/1508036684810.png)

现在我们把三步、四步、……、n步TD算法统称为n-step TD算法。我们可以把单步TD算法和蒙特卡洛算法看作是n步TD算法的特殊形式，n分别对应1和整个奖励序列的长度。

更正式的说，考虑这样一个序列$S_t,R_{t+1},S_{t+1},R_{t+2},\dots,R_T,S_T$，蒙特卡洛算法可以写成$v_{\pi}(S_t) = G_t = R_{t+1} + \gamma R_{t+2} + \dots + {\gamma} ^ {T-t-1} R_T$。

在单步返回中，值的估计是第一个奖励加上折扣乘以下一个状态的价值，公式为$G_{t:t+1} \approx  R_{t+1} + {\gamma} {V_t(S_{t+1})}$。

现在我们可以给出两步TD算法的更新公式可以写成$G_{t:t+2} \approx R_{t+1} + {\gamma} { R_{t+2}} + {\gamma} ^2 {V_{t+1}} (S_{t+2})$。

类似的，我们可以给出n-step TD算法的更新公式：

$G_{t:t+n} \approx R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^{n-1} R_{t+n} + \gamma^n V{t+n-1}(S_{t+n}) \tag{7.1}$

公式(7.1)可以看作是接受n个奖励，剩下的奖励由$V_{t+n-1}(S_{t+n})$代替。如果$t+n > T$，那么缺失项的值设为0，对于该状态，它的值的估计按照蒙特卡洛算法进行计算。

那么，这个n-step TD算法的公式为：

$$V_{t+n}(S_t) \approx V_{t+n-1}(S_t) + \alpha[G_{t:t+n} - V_{t+n-1}(S_t)], \ \ \ \ 0 \le t \le T \tag{7.2}$$

n-step TD算法代码如下：

![1508051918331](images/1508051918331.png)

### Exercise 7.1

问：根据式(7.2)，写出n-step TD算法的误差表示

答：

$$\begin{align}G_{t} - V{(S_t)} &= R_{t+1} + \dots + \gamma^{n-1}R_{t+n} + \gamma^{n}G_{t+n}    -V(S_t) + \gamma^n V(S_{t+n}) - \gamma^n V(S_{t+n}) \\ &= R_{t+1} + \dots + \gamma^{n-1}R_{t+n}      + \gamma^n V(S_{t+n}) - V(S_t) + \gamma^n G_{t+n}  - \gamma^n V(S_{t+n}) \\ &= \delta_t + \gamma^{n}\delta_{t+2n} + \dots + 0 \\ &= \sum_{i = 0}^{[len(T) / n]} {\gamma}^{n i}{\delta_{t + 2ni}} \end{align}$$

### Example 7.1: n-step TD Methods on the Random Walk

我们在上一章例6.2介绍了随机游走这个例子。现在我们使用n-step TD算法来解决这个问题。以2步TD算法为例，它将使$V(D)$和$V(E)$趋向于1。如果是3步TD算法，它将使$V(C)$ 、$V(D)$、$V(E)$趋于1。

n取多大最好？下图显示了在更大规模随机游走实验中，n的取值对实验结果的影响。在这个更大的随机游走实验中，总共有19个状态，到达最左边的奖励为-1，剩下的奖励全为0。通过对一些n和$\alpha$的取值，我们得到了如下图所示的实验结果。测量的依据是真实值和预测值之间的平均根方差。从图我们可以看到，蒙特卡洛算法和单步TD算法其实是n-step TD算法的极端情况。

![1508052451051](images/1508052451051.png)

### Exercise 7.3

问：为什么在本章中，我们要将状态由5个变为19个？小的学习率对于不同的n有什么区别？在较大的步伐中，左侧0到-1的结果有什么改变？你是怎样认为最优的n值？

答：如果状态还是5个，那么n=6和n=7的实验结果没有什么区别，都可以看作是蒙特卡洛算法。

$\alpha$较小时，对n较大时有利；n越小，$\alpha$越大，实验效果越好。

可能会更接近于-1。

n的最优取值是一个超参数，需要我们自己去调试。

## n-step Sarsa

如何将n-step TD预测算法应用到控制算法中？我们首先介绍n-step Sarsa。

我们依然使用行为状态对和$\epsilon-greedy$算法对Sarsa算法进行控制。如上所示，类似的，更新公式可以写成：

$$G_{t:t+n} \approx R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^n Q_{t+n-1}(S_{t+n},A_{t+n}), \ \ \ \ n \ge1, \ 0 \le t \le T-n \tag{7.4}$$

如果$t+n \ge T$，那么$G_{t:t+n} \approx G_t$。Sarsa算法可以写成：

$$Q_{t+n} (S_t,A_t) \approx Q_{t+n-1}(S_t,A_t) + \alpha [G_{t:t+n} - Q_{t+n-1} (S_t,A_t], \ \ \ \ {0} {\le} {t} {<} {T}  \tag{7.5}$$

上式就是整个n-step Sarsa算法。

n-step Sarsa算法的表示图可以为：

![1508054085730](images/1508054085730.png)

n-step Sarsa的伪代码如下图所示。

![1508054175434](images/1508054175434.png)

期望Sarsa如何计算？期望Sarsa与n-step Sarsa一致，除了最后一项不是确定的行为状态对，而是行为状态对的期望。在$n \ge 1 \ and \ 0 \le t \le T-n$下，它的更新公式如下：

$$G_{t:t+n} \approx R_{t+1} + \dots + \gamma ^{n-1} R_{t+n} + \gamma ^n \sum_a \pi (a|S_{t+n}) Q_{t+n-1}(S_{t+n},a) \tag{7.6}$$

![1508056349206](images/1508056349206.png)

## n-step Off-policy Learning by Importance Sampling

回忆一下以前的异策略，策略$b$是行为策略，它更多的是对当前状态的探索；策略$\pi$是目标策略，它更多的是贪婪的选择当前状态最大值对应的行为。我们用重要采样率将行为策略和目标策略联系起来。对于n-step算法，它可以写成：

$$V_{t+n}(S_t) \approx V_{t+n-1}(S_t) + {\alpha} \rho_{t:t+n-1} [G_{t:t+n} - V_{t+n-1}(S_t)] \ \ \ \ 0\le t \le T \tag{7.7}$$

这里$\rho_{t:t+n-1}$被称为重要采样率，它的计算公式为：

$$\rho_{t:h} \approx \Pi_{k=t}^{\min(h,T-1)}  \frac{\pi(A_k|S_k)} {b (A_k| S_k)} \tag{7.8}$$

重要采样率可以这样理解：如果某一个行为不会被策略$\pi$采用，那么$\rho = 0$；如果某一行为在$\pi$中选择的概率要比b大，那么我们就可以认为在$\pi$中该状态行为对的价值要比在b中大；如果某一行为在$\pi$中选择的概率要比b小，那么我们就可以认为在$\pi$中该状态行为对的价值要比在b中小。

在加上重要采样率后，上一节的Sarsa算法可以写成：

$$Q_{t+n}(S_t,A_t) \approx Q_{t+n-1}(S_t,A_t) + \alpha \rho_{t+1:t+n-1} [G_{t:t+n} - Q_{t+n-1}(S_t,A_t)] \tag{7.9}$$

现在，我们不必关心我们是如何的选取行为，我们关心的是我们能够从中学习到什么。伪代码如下：

![1508058255524](images/1508058255524.png)

n-step的期望Sarsa使用和Sarsa相同的更新方法，除了期望Sarsa的重要采样率要比Sarsa少一个因子。我们需要将$\rho_{t+1:t+n-2}$代替$\rho_{t+1:t+n-1}$。减少一个因子的原因是，期望Sarsa在最后要考虑所有的行为选择情况，因此不需要进行修正。

## Per-reward Off-policy Methods

在上一节提出的多步异策略方法概念非常清晰，也非常简单，但它可能不是最有效的方法。一个更复杂的方法是在每步都乘以比率，正如我们在章节5.9所叙述的那样。为了了解这个方法，我们先来看一下一般的多步TD算法，它可以递归的写成$G_{t:h} = R_{t+1} + \gamma G_{t+1:h}$。

对于所有的结果经验，包括第一个返回值$R_{t+1}$和下一个状态$S_{t+1}$在内，都要在每个时间片t乘上重要采样率$\rho_t = \frac{\pi(A_t|S_t)} {b(A_t|S_t)}$。假设在t时刻的行为被策略$\pi$选择的概率为0，那么$\rho$就等于0，这样n步返回值的结果就是0。如果把它当作是趋近的目标，那么就会产生非常大的方差。所以，我们可以将式子变换一下：

$$G_{t:h} = \rho_t(R_{t+1} + \gamma G_{t+1:h}) + (1 - \rho_t)V(S_t), \ \ \ \ t < h \le T \tag{7.10}$$

现在，如果$\rho_t$等于0，目标值就和估计值相同，而不是目标值变成0导致估计值变小。重要采样率变为0意味着我们应该忽略这个样本，所以让估计值保持不变似乎是一个正确的结果。重要采样率的期望值是1，这说明它与估计值不相关，因此第二项的期望值应该是0。

在同策略的情况下，$\rho_t$总是1。

### Exercise 7.4

问：上述过程的伪代码

答：

Input: an arbitrary behavior policy b such that b(a|s) > 0, $ \forall s \in S$,$a \in A$

Initialize $Q(s,a)$ arbitrarily, $\forall s \in S$,$a \in A$

Initialize $\pi$ to be $\epsilon-greedy$ with respect to Q, or as a fixed given policy

Parameters: step size $\alpha \in (0,1]$, small $\epsilon > 0$, a positive integer n

All stroe and access operations (for $S_t$, $A_t$, and $R_t$) can take their index mod n

Repeat(for each episode):

​	Initialzie and store $S_0 \neq$ terminal

​	Select and store an action $A_0 \sim b(\cdot|S_0)$

​	T $\gets$ $\infty$

​	For t = 0,1,2,...:

​		If t < T, then:

​			Take action $A_t$

​			Observe and store the next reward as $R_{t+1}$ and the next state as $S_{t+1}$

​			If $S_{t+1}$ is terminal, then:

​				T $\gets$ t + 1

​			else:

​				Select and store an action $A_{t+1} \sim b(\cdot|S_{t+1})$

​		$\tau \gets t - n + 1$ ($\tau$ is the time whose estimate is being updated)

​		if $\tau \ge 0$:

​			$\rho \gets \prod^{min(\tau + n -1, T-1)}_{i = \tau +1} \frac{\pi(A_i|S_i)} {b(A_i|S_i)}$

​			$G \gets \sum^{min(\rho+n,T)}_{i = \tau + 1} \gamma^{i - \tau-1}R_{i}$

​			If $tau + n < T$,then:

​				$G \gets \rho_t(R_{t+1} + \gamma G_{t+1:h}) + (1 - \rho) V(S_t)$

​			$Q_(S_{\tau}, A_{\tau}) \gets Q(S_{\tau},A_{\tau}) + \alpha \rho [G-Q(S_{\tau}, A_{\tau})]$

​			If $\pi$ is being learned, then ensure that $\pi{\cdot| S_{\tau}}$ is $\epsilon-greedy$ wrt Q

​	Until $\tau = T - 1$

对于行为价值，多步返回值的异策略的定义与期望Sarsa一致。但不同之处在于，第一个行为并没有在重要采样中起到作用。我们只关心从行为中学习到的值，不关心甚至不考虑该行为是否在目标策略中出现。对于现在的奖励和行为，我们设权重为1，后面的行为才考虑重要采样率。多步返回的异策略迭代定义可以写成：

$$G_{t:h} \approx R_{t+1} + \gamma (\rho_{t+1}G_{t+1:h} + (1-\rho_{t-1}) \bar{Q}_{t+1}) \tag{7.11}$$

我们在本节、以前的章节和第五章所采用的重要性采样都可以应用到异策略学习中，代价就是增大更新的方差。增大的方差需要我们采用一个小的学习率来减缓学习速度。这也是异策略训练比同策略学习慢的原因。毕竟，我们得到的数据与我们尝试学习到的策略的关联很少。

