import numpy as np
import matplotlib.pyplot as plt

class Bernoulli:
    """ Bernoulli Arm """

    def __init__(self,theta):
        # create a Bernoulli arm with mean theta
        self.mean = theta
        self.variance = theta*(1-theta)

    def sample(self):
        # generate a reward from a Bernoulli arm 
        return float(np.random.rand() < self.mean)


class Gaussian:
    """ Gaussian Arm """

    def __init__(self,mu,var=1):
        # create a Gaussian arm with specified mean and variance
        self.mean = mu
        self.variance = var

    def sample(self):
        # generate a reward from a Gaussian arm 
        return self.mean + sqrt(self.variance)*np.random.randn()
        

class Exponential:
    """ Exponential Arm """

    def __init__(self,p):
        # create an Exponential arm with parameter p
        self.mean = 1/p
        self.variance = 1/(p*p)

    def sample(self):
        # generate a reward from an Exponential arm 
        return -(self.mean)*np.log(np.random.rand())

class TruncatedExponential:
    """ Truncated Exponential Arm """

    def __init__(self,p,trunc):
        # create a truncated Exponential arm with parameter p
        self.p = p
        self.trunc = trunc
        self.mean = (1.-np.exp(-p * trunc)) / p
        self.variance=0
        
    def sample(self):
        # generate a reward from an Exponential arm 
        return min(-(1/self.p)*np.log(np.random.rand()),self.trunc)


class MixedMAB():
    """ Mixed-Arm Multi-Arm Bandit Machine """

    def __init__(self,arms):
        """given a list of arms, create the MAB environnement"""
        self.arms = arms
        self.n_arms = len(arms)
        self.means = [arm.mean for arm in arms]
        self.bestarm = np.argmax(self.means)
    
    def rwd(self,a):
        return self.arms[a].sample()


class GaussianMAB():
    """ Gaussian-Arm Multi-Arm Bandit Machine """

    def __init__(self,n_arms=2,labels=None,means=None):
        self.n_arms = n_arms
        if means is not None:
            self.means = means
        else:
            self.means = 0 + np.random.randn(self.n_arms)
        self.sigma = np.random.rand(self.n_arms) * 1
        if labels is None:
            labels = np.arange(n_arms)

    def rwd(self,a):
        return np.random.randn() * self.sigma[a] + self.means[a]
        
    def draw(self, ax): 
        dist = []
        for a in range(self.n_arms):
            dist.append(np.random.normal(self.means[a], self.sigma[a], int(5000)))
        bp = ax.violinplot(dist)


class BernoulliMAB():
    """ Bernoulli-Arm Multi-Arm Bandit Machine """

    def __init__(self,n_arms=2,labels=None,means=None):
        self.n_arms = n_arms
        if means is None:
            self.means = np.random.rand(self.n_arms)
        else:
            self.means = np.array(means)
        if labels is None:
            labels = np.arange(n_arms)
        self.labels = labels

    def rwd(self,a):
        return int(np.random.rand() < self.means[a]) 

    def draw(self, ax): 
        ax.set_title("Bernoulli Bandit")
        ax.set_ylim([0,1])
        #dist = []
        #for a in range(self.n_arms):
        #    dist.append(np.random.normal(self.means[a], self.sigma[a], int(5000)))
        #bp = ax.violinplot(dist)


def draw(env, policy, info=None, ax=None):

    if ax is None:
        fig = plt.figure(figsize=[4,3])
        ax = fig.add_subplot(111)

    ax.set_xlabel("$A$")
    ax.set_ylabel("$R$")
    ax.scatter(np.arange(env.n_arms)+1,env.means,marker='*',color='gray',s=50,label=r"$\mu_*$") 
    ax.set_xticks(np.arange(env.n_arms)+1, ['%d' % (_+1) for _ in range(env.n_arms)])
    labels=np.arange(env.n_arms)+1
    if info is not None and 'labels' in info.keys():
        labels=info['labels']
    ax.set_xticklabels(labels)

    # Bandit Env - specific drawing
    if hasattr(env, 'draw'):
        env.draw(ax)
    # Bandit Alg - specific drawing
    if hasattr(policy, 'draw'):
        policy.draw(ax)
