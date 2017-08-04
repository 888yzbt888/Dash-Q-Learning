#import Dash_simulator as dash
#import random
#import numpy as np
#maxT=100
#t=0
#sim=dash.Client()
#while t<maxT:
#    a=np.zeros([6])
#    b=random.randrange(0,6,1)
#    a[b]=1
#    buf,BW,reward,timestep=sim.frame_step(a)
#    t=timestep
#    print('buf:',buf,' BW:',BW,' reward:',reward,' timestep:',timestep,' b:',b+1)

import tensorflow as tf
import Dash_simulator as dash
import random#
import numpy as np
from collections import deque

ACTIONS=7
GAMMA=0.1
ALPHA=0.1
OBSERVE=100000
EXPLORE=2000000
EPSILON=0.1#frequency adjusted
REPLAY_MEMORY=50000
BATCH=32

#def weight_variable(shape):
#    initial=tf.truncated_normal(shape,stddev=0.01)
#    return tf.Variable(initial)

#def bias_variable(shape):
#    initial=tf.Variable(initial)

def createQtable():
    #Qtable=tf.placeholder('float',[10,7,7])
    Qtable=np.zeros([10,7,7])
    return Qtable

def ip2op(Qtable):
    ip=tf.placeholder('float',[2])
    index=np.argmax(Qtable[ip[0],ip[1],:])
    op=np.zeros([7])
    op[index]=1
    return ip,op

def train(Q,sess):
    #a=tf.placeholder('float',[None,ACTIONS])
    #y=tf.placeholder('float',[None])
    #ac=tf.reduce_sum(tf.multiply(op,a),reduction_indices=1)
    #cost=tf.reduce_mean(tf.square(y=readout_action))
    #train_step=tf.train.AdamOptimizer(1e-6).minimize(cost)

    net_state=dash.Client()

    D=deque()

    firstAction=np.zeros(ACTIONS)
    firstAction[random.randrange(0,ACTIONS)]=1
    buf,BW,reward,t=net_state.frame_step(firstAction)

    #saver=?
    #sess.run(tf.initialize_all_variables())

    while t<OBSERVE:
        a_t=np.zeros([ACTIONS])
        if random.random()<=EPSILON:
            a_t[random.randrange(ACTIONS)]=1
            print('random action')
        else:
            a_t[np.argmax(Q[buf,BW,:])]=1

        #scale down epsilon

        Lbuf=buf
        LBW=BW
        buf,BW,reward,t=net_state.frame_step(a_t)
        Q[int(Lbuf),int(LBW),np.argmax(a_t)]=Q[int(Lbuf),int(LBW),np.argmax(a_t)]+ALPHA*(reward+GAMMA*np.amax(Q[int(buf),int(BW),:])-Q[int(Lbuf),int(LBW),np.argmax(a_t)])
        
        

def runSim():
    sess=tf.InteractiveSession()
    Q=createQtable()
    train(Q,sess)
    #? ? ?=createnetwork()
    #train(?,?,?,sess)

runSim()
