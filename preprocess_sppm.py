import os
import shutil
import numpy as np

import itertools
import multiprocessing
from multiprocessing import Pool
from functools import partial

# Rename from fullname to DC form
for dirpath, dirnames, filenames in os.walk('./dataset'):
    print(dirpath)
    print(len(filenames))
    currScene = ""
    currIter = ""
    cnt = 0
    for idx, filename in enumerate(sorted(filenames)):
        if filename.split('_')[0].isnumeric():
            continue

        # Parse scene and iterations
        sceneName = filename.split('_')[0]
        iterations = filename.split('_')[1]
        bufIdx = int(filename.split('_')[4][0])
        if currScene == "" and currIter == "":
            currScene = sceneName
            currIter = iterations
        elif currScene != sceneName or currIter != iterations:
            currScene = sceneName
            currIter = iterations
            cnt = 0
        method = "sppm"

        split = filename.split('_')[2:]

        if cnt != 0 and cnt % 5 == 0 and bufIdx == 3:
            cnt += 5

        # New filename
        num = cnt
        if bufIdx == 3:
            cnt += 1
        split[0] = '%03d' % (num % 50)
        newname = '_'.join(split)
        print(newname)

        # Make directory
        directory = os.path.join(dirpath, "_".join([sceneName, method, iterations]))
        os.makedirs(directory, exist_ok=True)
        
        # Move file
        fullpath =  os.path.join(directory, newname)
        os.rename(os.path.join(dirpath, filename), fullpath)
