import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from scipy.stats import norm

A_LEFT = 1
A_RIGHT = 2
A_UP = 3
A_DOWN = 4

class MouseGrid(gym.Env):
    '''
        A gridworld MDP. 
    '''

    def __init__(self,version='V3'):

        # Grid representation
        self.G = np.array([[0,0,0,0,1,0],
                           [0,0,1,0,1,0],
                           [0,0,1,0,0,0],
                           [0,0,0,0,0,0]])
        self.n_states = self.G.size
        self.L=np.arange(self.n_states).reshape(self.G.shape).astype(int)

        # Action space
        self.action_space = gym.spaces.Discrete(self.n_states)
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.int8,
        )

        # Emmision probabilities o ~ N( . | mu[s], sig[s])
        # (TODO: it makes sense to call this self.mu)
        self.P = (1 - self.G.flatten()) * np.random.rand(self.G.size) * 1

        if version == 'V4':
            # TODO generate such that self.P[s] correlated with dist(s, s_start)
            pass

        # Transition probabilities 
        def generate_transitions(G):
            '''
                Returns
                -------
                Action set (available actions) A
                    where A[s] = A(s)
                Transition probabilities T
                    where T[s] = P(S | s, policy(s))
                    according to policy(s) -- by default, uniformly random
            '''
            n = G.size
            T = np.zeros((n,n))
            A = [[] for _ in range(n)]
            for i in range(G.shape[0]):
                for j in range(G.shape[1]):
                    # Current cell index
                    s = i * G.shape[1] + j

                    # 0. We can stay where we are
                    #T[s,s] = 1
                    #A[s].append(s)

                    # 1. Check up (i-1, j)
                    if i > 0 and G[i-1, j] != 1:
                        T[s,(i-1) * G.shape[1] + j] = 1
                        A[s].append(A_UP)
                    
                    # 2. Check down (i+1, j)  
                    if i < G.shape[0] - 1 and G[i+1, j] != 1:
                        T[s,(i+1) * G.shape[1] + j] = 1
                        A[s].append(A_DOWN)
                    
                    # 3. Check left (i, j-1)
                    if j > 0 and G[i, j-1] != 1:
                        T[s,i * G.shape[1] + (j-1)] = 1
                        A[s].append(A_LEFT)
                    
                    # 4. Check right (i, j+1)
                    if j < G.shape[1] - 1 and G[i, j+1] != 1:
                        T[s,i * G.shape[1] + (j+1)] = 1
                        A[s].append(A_RIGHT)

                    T[s,:] = T[s,:] / np.sum(T[s,:])

            return T, A

        # NOTE: self.T only provides viable next states, even if later we 
        #       suppose that that P(each of those states) is equal/uniform
        #       -- this would not be the case if a non-random policy was used!
        self.T, self.A = generate_transitions(self.G)

        # Defaults
        self.s_start = 0
        self.s_goal = 5

        # Reset
        self.reset()

    def tile2cell(self, s):
        return divmod(s, self.G.shape[1]) 

    def cell2tile(self, cell):
        return int(cell[0] * self.G.shape[1] + cell[1])

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.s = self.s_start
        o =  self.P[self.s] + np.random.randn() * 0.1
        #o = int(np.random.rand() > self.P[self.s])
        return o, {}

    def f(self,s,a):
        ''' transition fn (dynamics fn) of the environment '''

        i,j = self.tile2cell(s)

        if a == 0:  # nothing
            pass
        elif a == A_LEFT:  # left
            j = max(0, j - 1)
        elif a == A_RIGHT:  # right
            j = min(self.G.shape[1] - 1, j + 1)
        elif a == A_UP:  # up
            i = max(0, i - 1)
        elif a == A_DOWN:  # down
            i = min(self.G.shape[0] - 1, i + 1)

        return self.cell2tile((i,j))

    def step(self,a):
        ''' Step function. 
        '''

        self.a = a

        # transition
        self.s = self.f(self.s,a)

        # reward
        r = int(self.s == self.s_goal)

        # emission
        o = self.P[self.s] + np.random.randn() * 0.1
        #o = int(np.random.rand() > self.P[self.s])

        return o, r, bool(r > 0), False, {}

    def render(self, mode='human', ax=None):
        if mode == 'empty':
            draw(self.G, ax=ax, L=np.arange(self.n_states).reshape(self.G.shape).astype(int))
            #draw(proc.world.G, s=None, a=None, o=o, ax=ax, )
        elif mode == 'probabilities':
            draw(self.P.reshape(self.G.shape), ax=ax, L=np.arange(self.n_states).reshape(self.G.shape).astype(int))
        elif mode == 'policy':
            draw(self.T[self.s].reshape(self.G.shape), ax=ax, L=np.arange(self.n_states).reshape(self.G.shape).astype(int))
            #draw(proc.world.T[3].reshape(proc.world.G.shape), s=3, a=None, o=o, ax=ax, L=np.arange(proc.world.n_states).reshape(proc.world.G.shape).astype(int))
        else:
            print("Mode not recognised")

    def close(self):
        pass

def draw(G, s=None, a=None, ax=None, x_labels=None, y_labels=None, L=None, s_est=None):
    n,m = G.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=[m, n])

    else:
        ax.clear()

    im = ax.imshow(G, cmap=plt.cm.Grays, vmin=0, vmax=1)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(0, m, 1))
    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    if x_labels is not None:
        ax.set_xticklabels(x_labels)

    ax.set_yticks(np.arange(0, n, 1))
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    #ax.set_yticklabels(np.arange(0, n, 1))
    if y_labels is not None:
        ax.set_yticklabels(y_labels)

    ax.grid(which='minor', color='k')

    if L is not None:
        for i in range(n):
            for j in range(m):
                ax.text(j, i, f"{L[i,j]}", va='center', ha='center', color='black', fontsize=10)

    if s_est is not None:
        i, j = divmod(s_est, G.shape[1]) 
        ax.text(j, i, r'$\hat s$', va='center', ha='center', color='red', fontsize=10)   


    if s is not None:
        i, j = divmod(s, G.shape[1]) 
        icon = r"$\diamond$"
        if a is not None:
            sym = [r"$o$", r"$\triangleleft$", r"$\triangleright$", r"$\bigtriangleup$", r"$\bigtriangledown$"]
            icon = sym[a]
        ax.text(j, i, icon, va='center', ha='center', color='blue', fontsize=50)
        #if o is not None:
        #    ax.text(j, i, r'$*$', va='center', ha='center', color='red', fontsize=10)   

    return ax


class MouseProcess():
    '''
        A POMDP. By wrapping around a MDP with a random policy. 
    '''

    def __init__(self):

        self.env = MouseGrid()
        self.state_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.policy = np.random.choice
        #self.P = self.env.P
        #self.T = self.env.T
        self.n_states = self.env.n_states

        self.reset()

    def reset(self, seed=None, options=None):
        ''' Reset the environment '''
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        return self.env.reset(seed, options)

    def P_S(self,s=None):
        ''' The pmf of the next state, given current state 's'. 
        P( S[t] | S[t-1] = s) 
        so, evaluating the pm of the next state P_S(s)[s_] gives:
        P( S[t] = s_ | S[t-1] = s)

        NOTE: self.env.T only provides viable next states, here we  
              suppose that that P(each of those states) is equal/uniform
              -- this would not be the case for a non-random mouse!
        '''
        return self.env.T[s,:]

    def P_O(self,o,s):
        ''' Evaluates the pd of the observation o, given current state 's'. 
        P( O[t] = o | S[t] = s) 
        '''
        return norm.pdf(o, loc=self.env.P[s], scale=0.1)

    def render(self, mode):
#        ''' Draw the current state of affairs. '''
        return self.env.render(mode)

    def step(self, pred=None):
        ''' Generate the next state, and return its observation. '''
        a = self.policy(self.env.A[self.env.s])
        o, r, terminated, truncated, _ = self.env.step(a)
        return o, terminated, truncated, _

    def close(self):
        pass

if __name__ == "__main__":

    # Instantiate the POMP 
    proc = MouseProcess()

    # Generate trajectory (and draw/animate) of observations.
    o, _ = proc.reset()
    o_seq = [o]
    plt.ion()
    fig, ax = plt.subplots(figsize=[8, 6])
    draw(proc.env.G, s=proc.env.s, a=None, ax=ax, L=proc.env.L)
    for t in range(10):
        o, terminated, truncated, _ = proc.step()
        o_seq.append(o)
        plt.pause(0.5)  
        draw(proc.env.G, s=proc.env.s, a=proc.env.a, ax=ax, L=proc.env.L)
    plt.ioff()

    # Close the generating process
    proc.close()
