# main automation script. Calls other automation scripts.
# zynet seems to be python framework which automates NN creation for fpga. However, I found there is a zynet.v file as well so I am not sure why do we need a verilog file for it. I understand that the zynet is actually automating the hardware side of things. Hence, the framework internally calls verilog files.
# I feel that this file has to be run after the python training is complete as it needs the WeightsAndBiases.txt file.
# I am not sure but it seems the author himself came up with this framework. Seems like tremendous work.

from zynet import zynet
from zynet import utils
import numpy as np

def genMnistZynet(dataWidth,sigmoidSize,weightIntSize,inputIntSize):
    model = zynet.model()
    model.add(zynet.layer("flatten",784))
    model.add(zynet.layer("Dense",30,"sigmoid"))
    model.add(zynet.layer("Dense",20,"sigmoid"))
    model.add(zynet.layer("Dense",10,"sigmoid"))
    model.add(zynet.layer("Dense",10,"hardmax"))
    weightArray = utils.genWeightArray('WeigntsAndBiases.txt')
    biasArray = utils.genBiasArray('WeigntsAndBiases.txt')
    model.compile(pretrained='Yes',weights=weightArray,biases=biasArray,dataWidth=dataWidth,weightIntSize=weightIntSize,inputIntSize=inputIntSize,sigmoidSize=sigmoidSize)
    zynet.makeXilinxProject('myProject1','xc7z020clg484-1')
    zynet.makeIP('myProject1')
    zynet.makeSystem('myProject1','myBlock2')
    
if __name__ == "__main__":
    genMnistZynet(dataWidth=8,sigmoidSize=10,weightIntSize=4,inputIntSize=1)