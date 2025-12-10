#!/usr/bin/env python3

import random
import numpy
import sys
import time

problem = {
    "numAgents" : 2,
    "Dimension" : (5, 5),
    "Density" : 0.20
}

def problemCreate(file):
    sizeX = problem["Dimension"][0]
    sizeY = problem["Dimension"][1]
    
    # header
    file.write(str(sizeX) + " " + str(sizeY) + "\n")

    # problem space
    space = numpy.zeros((sizeX, sizeY))

    for x in range(0, sizeX):
        for y in range(0, sizeY):
            if (random.uniform(0, 1) < problem["Density"]):
                file.write("@ ")
                space[x, y] = 1
            else:
                file.write(". ")
                space[x, y] = 0
        file.write("\n")
             
    
    # agent header
    file.write(str(problem["numAgents"]) + "\n")

    # agent placement
    for i in range(0, problem["numAgents"]):
        # initial agent positions
        aX = int(random.uniform(0, sizeX))
        aY = int(random.uniform(0, sizeY))
        # intial goal positions
        pX = int(random.uniform(0, sizeX))
        pY = int(random.uniform(0, sizeY))


        timeout = (sizeX*sizeY/1000)
        start = time.time()
        while (space[aX, aY] != 0) and (time.time() - start < timeout):
            aX = int(random.uniform(0, sizeX))
            aY = int(random.uniform(0, sizeY))
        space[aX, aY] = 2

        start = time.time()
        while (space[pX, pY] != 0) and (time.time() - start < timeout):
            pX = int(random.uniform(0, sizeX))
            pY = int(random.uniform(0, sizeY))
        space[pX, pY] = 2

        file.write("{} {} {} {}\n".format(aX, aY, pX, pY) )

        pass
    pass

if __name__ == '__main__':
    problem["numAgents"] = int(sys.argv[1])
    problem["Dimension"] = (int(sys.argv[2]), int(sys.argv[3]))
    problem["Density"] = float(sys.argv[4])
    numberOfProblems = int(sys.argv[5]) #+ 1
    # for i in range(1,numberOfProblems): # removed and replaced into bash
    with open("tmpInstances/exp" + str(numberOfProblems) + ".txt", "w") as f:
        problemCreate(f)

    f.close()