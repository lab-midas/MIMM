from datetime import datetime
import os
def loadPaths(params_dict):
    trainType = params_dict["trainType"]["value"]
    savePath = os.path.expanduser('~')+r'/MIMM/src/SaveWandbRuns/'+trainType
    saveFile = savePath+datetime.now().strftime('/%H_%M_%d_%m_%Y.txt')
    syncFile = savePath+datetime.now().strftime('/sync_%H_%M_%d_%m_%Y.txt')

    if not os.path.exists(savePath):
        os.chdir(os.path.expanduser('~')+'/MIMM/src')
        os.makedirs(savePath)

    return saveFile, syncFile