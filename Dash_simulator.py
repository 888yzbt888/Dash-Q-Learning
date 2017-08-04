#基本完成，未发现明显bug
import numpy as np
import random
import math

bufR=-1
stopR=-100
qualityR=-1
switchR=-1
changeProb=0.005
maxBW=7
class Client:
    def __init__(self):
        self.maxBuf=10
        self.buf=0
        self.timestep=0
        self.downloaded=0
        self.stored=0
        self.BW=random.randrange(0,maxBW,1)
        self.lastQ=self.BW
        
    def frame_step(self,input_actions):
        reward=0
        self.BW=getNetworkState(self.BW)
        print('BW:',self.BW+1)
        if sum(input_actions)!=1:
            raise ValueError('Multiple input acitons!')
        quality=np.argmax(input_actions)
        print('quality:',quality+1)
        self.downloaded+=1
        print('downloaded:',self.downloaded)
        deltaT=(quality+1)/(self.BW+1)
        print('deltaT=',deltaT)
        self.timestep+=deltaT
        #if self.timestep==float('infinity'):
        #    self.timestep=1000000
        self.buf=math.ceil(self.downloaded-int(self.timestep))
        if self.buf==self.maxBuf:
            self.timestep+=1
            self.buf-=1
        if self.buf<=0:
            self.buf=0
            reward+=stopR
            self.downloaded=self.timestep
        else:
            reward+=bufR*(self.maxBuf-self.buf)
        print('buf:',self.buf)
        print('timestep:',self.timestep)
        reward+=qualityR*(maxBW-quality)
        reward+=switchR*abs(self.lastQ-quality)
        print('reward',reward,'\n=========')
        self.lastQ=quality
        return self.buf,self.BW,reward,self.timestep
    
def getNetworkState(BW):
    ran=random.random()
    if ran>changeProb:
        BW=BW
    else:
        BW=random.randrange(0,maxBW,1)
    return BW
    
