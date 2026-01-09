import numpy as np
import time

def evaluate(proc,f,N,T):
    '''
    '''
    returns = []
    times = []
    for i in range(N):

        o, _ = proc.reset()
        o_seq = [o]
        s = proc.env.s
        r = float(f(proc,o_seq) == proc.env.s)

        t = 0
        end = False
        while not end:
            t += 1
            o, terminated, truncated, _ = proc.step()
            o_seq.append(o)
            s = proc.env.s
            start = time.perf_counter()
            s_pred, s_probs = f(proc,o_seq)
            times.append(time.perf_counter() - start)
            #print(s_pred,proc.env.s)
            r += int(s_pred == proc.env.s)
            end = terminated or truncated or t >= T

        returns.append(r)

    return { 
            "average reward per episode" : float(np.mean(returns)), 
            "average time per estimator call" : float(np.mean(times)),
            }

