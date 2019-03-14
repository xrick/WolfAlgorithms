
import json
import os
import sys

class HMM(object):

    def __init__(self,configFile):
        if configFile == None:
            print("請載入設定檔再進行！")
            sys.exit()
        self.model = json.loads(open("iniData/"+configFile).read())["hmm"]
        self.StateTransitionProbability = self.model["A"] #狀態轉移機率
        self.MarkovProcessStates = self.StateTransitionProbability.keys()  # markov model狀態
        self.NumOfStates = len(self.MarkovProcessStates)  # markov model狀態的數目
        self.EmissionProbability = self.model["B"] #observation probabilities
        self.Symbols = self.EmissionProbability.values()[0].keys #可觀察符號
        self.NumOfSymbols = len(self.Symbols)#可觀察符號的數目
        self.pi = 0 #初始狀態
        

    def forward(self,obs):
        self.fwd = [{}]


    def backward(self,obs):
        pass


if __name__ == "__main__":
    hmm = HMM("random1.json")
    print(hmm.model)
    
