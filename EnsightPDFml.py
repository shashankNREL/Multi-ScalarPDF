#!/usr/bin/env python3

import numpy as np
import itertools
from scipy import stats
import pandas as pd
import os
import argparse
import time
from datetime import timedelta


def FillPeriodic(Orig, nx, width):
    Mod = np.zeros((nx+2*width, nx+2*width, nx+2*width))

    Mod[width:nx+width, width:nx+width, width:nx+width] = Orig[:, :, :]

    Mod[0:width, width:nx+width, width:nx+width] = Orig[nx-width:nx, :, :]
    Mod[(nx+width):(nx+2*width), width:nx+width,
        width:nx+width] = Orig[0:width, :, :]

    Mod[:, 0:width, :] = Mod[:, nx:(nx+width), :]
    Mod[:, (nx+width):(nx+2*width), :] = Mod[:, width:2*width, :]

    Mod[:, :, 0:width] = Mod[:, :, nx:(nx+width)]
    Mod[:, :, (nx+width):(nx+2*width)] = Mod[:, :, width:2*width]

    return Mod


def ReadEnsight(fileScalarA, fileScalarB, nx, width=32, npx=8):

    # number of points in each direction for indiviudal processors
    nx_ = nx//npx
    with open(fileScalarA, "rb") as f:
        varName = f.read(80)
        PartName = f.read(80)
        np.fromfile(f, dtype=np.int32, count=1)
        f.read(80)
        ScalarA = np.fromfile(f, dtype=np.float32, count=-1)
    f.close()
    ScalarA = np.reshape(ScalarA, (nx_, nx_, nx_, npx, npx, npx), order='F')
    ScalarAreshape = np.zeros((nx, nx, nx))
    for ip in range(npx):
        for jp in range(npx):
            for kp in range(npx):
                ScalarAreshape[ip*nx_:(ip+1)*nx_,
                               jp*nx_:(jp+1)*nx_,
                               kp*nx_:(kp+1)*nx_] = ScalarA[:,
                                                            :, :, kp, jp, ip]
    ScalarA = ScalarAreshape

    with open(fileScalarB, "rb") as f:
        varName = f.read(80)
        PartName = f.read(80)
        np.fromfile(f, dtype=np.int32, count=1)
        f.read(80)
        ScalarB = np.fromfile(f, dtype=np.float32, count=-1)
    f.close()
    ScalarB = np.reshape(ScalarB, (nx_, nx_, nx_, npx, npx, npx), order='F')
    ScalarBreshape = np.zeros((nx, nx, nx))
    for ip in range(npx):
        for jp in range(npx):
            for kp in range(npx):
                ScalarBreshape[ip*nx_:(ip+1)*nx_,
                               jp*nx_:(jp+1)*nx_,
                               kp*nx_:(kp+1)*nx_] = ScalarB[:, :, :,
                                                            kp, jp, ip]
    ScalarB = ScalarBreshape

    # Using Periodic BC, fill up the ghost cells = width wide on each side
    FullScalarA = FillPeriodic(ScalarA, nx, width)
    FullScalarB = FillPeriodic(ScalarB, nx, width)
    FullScalarC = np.subtract(np.subtract(1.0, FullScalarA), FullScalarB)

    return FullScalarA, FullScalarB, FullScalarC


def getBinCenter(nbinsA, nbinsB, Zst, nonUni=True):
    binsA = np.zeros(nbinsA)
    binsB = np.zeros(nbinsB)

    if(nonUni):
        zcutA = int(nbinsA/2)
        zcutB = int(nbinsB/2)

        dzA = Zst / float(zcutA-1)
        dzB = Zst / float(zcutB-1)

        for i in range(0, zcutA):
            binsA[i] = float(i) * dzA

        for i in range(0, zcutB):
            binsB[i] = float(i) * dzB

        m11 = float((nbinsA-1)**2-(zcutA-1)**2)
        m12 = float(nbinsA-zcutA)
        m21 = float(2*(zcutA-1)+1)
        m22 = 1.0
        r1 = 1.0 - Zst
        r2 = dzA
        delta = m11*m22-m12*m21
        a = (+ m22*r1 - m12*r2)/delta
        b = (- m21*r1 + m11*r2)/delta
        c = Zst - a*(zcutA-1)**2-b*(zcutA-1)
        for i in range(zcutA, nbinsA):
            binsA[i] = a*float(i)**2 + b*float(i) + c

        m11 = float((nbinsB-1)**2-(zcutB-1)**2)
        m12 = float(nbinsB-zcutB)
        m21 = float(2*(zcutB-1)+1)
        m22 = 1.0
        r1 = 1.0 - Zst
        r2 = dzB
        delta = m11*m22-m12*m21
        a = (+ m22*r1 - m12*r2)/delta
        b = (- m21*r1 + m11*r2)/delta
        c = Zst - a*(zcutB-1)**2-b*(zcutB-1)
        for i in range(zcutB, nbinsB):
            binsB[i] = a*float(i)**2 + b*float(i) + c
    else:
        binsA = np.linspace(0.0, 1.0, nbinsA)
        binsB = np.linspace(0.0, 1.0, nbinsB)

    Z1Z1 = np.zeros((nbinsA, nbinsB))
    Z2Z2 = np.zeros((nbinsA, nbinsB))
    Z1Z2 = np.zeros((nbinsA, nbinsB))
    Z1M = np.zeros((nbinsA, nbinsB))
    Z2M = np.zeros((nbinsA, nbinsB))

    for i in range(nbinsA):
        for j in range(nbinsB):
            Z1M[i, j] = binsA[i]
            Z2M[i, j] = binsB[j]
            Z1Z1[i, j] = binsA[i]*binsA[i]
            Z2Z2[i, j] = binsB[j]*binsB[j]
            Z1Z2[i, j] = binsA[i]*binsB[j]

    return Z1M, Z2M, Z1Z1, Z2Z2, Z1Z2


def genNonUniformBins(nbinsA, nbinsB, Zst, nonUni=True):

    binsA = np.zeros(nbinsA)
    binsB = np.zeros(nbinsB)

    if(nonUni):
        zcutA = int(nbinsA/2)
        zcutB = int(nbinsB/2)

        dzA = Zst / float(zcutA-1)
        dzB = Zst / float(zcutB-1)

        for i in range(0, zcutA):
            binsA[i] = float(i) * dzA

        for i in range(0, zcutB):
            binsB[i] = float(i) * dzB

        m11 = float((nbinsA-1)**2-(zcutA-1)**2)
        m12 = float(nbinsA-zcutA)
        m21 = float(2*(zcutA-1)+1)
        m22 = 1.0
        r1 = 1.0 - Zst
        r2 = dzA
        delta = m11*m22-m12*m21
        a = (+ m22*r1 - m12*r2)/delta
        b = (- m21*r1 + m11*r2)/delta
        c = Zst - a*(zcutA-1)**2-b*(zcutA-1)
        for i in range(zcutA, nbinsA):
            binsA[i] = a*float(i)**2 + b*float(i) + c

        m11 = float((nbinsB-1)**2-(zcutB-1)**2)
        m12 = float(nbinsB-zcutB)
        m21 = float(2*(zcutB-1)+1)
        m22 = 1.0
        r1 = 1.0 - Zst
        r2 = dzB
        delta = m11*m22-m12*m21
        a = (+ m22*r1 - m12*r2)/delta
        b = (- m21*r1 + m11*r2)/delta
        c = Zst - a*(zcutB-1)**2-b*(zcutB-1)
        for i in range(zcutB, nbinsB):
            binsB[i] = a*float(i)**2 + b*float(i) + c
    else:
        binsA = np.linspace(0.0, 1.0, nbinsA)
        binsB = np.linspace(0.0, 1.0, nbinsB)

    Za = np.zeros(len(binsA)+1)
    Zb = np.zeros(len(binsB)+1)
    dzAB = np.zeros((nbinsA, nbinsB))

    Za[0] = binsA[0]-(0.5*binsA[1]+0.5*binsA[0]-binsA[0])
    Zb[0] = binsB[0]-(0.5*binsB[1]+0.5*binsB[0]-binsB[0])

    Za[-1] = binsA[-1]+(binsA[-1]-0.5*binsA[len(binsA)-1]
                        - 0.5*binsA[len(binsA)-2])
    Zb[-1] = binsB[-1]+(binsB[-1]-0.5*binsB[len(binsB)-1]
                        - 0.5*binsB[len(binsB)-2])

    for j in range(1, len(binsA)):
        Za[j] = 0.5*binsA[j]+0.5*binsA[j-1]

    for j in range(1, len(binsB)):
        Zb[j] = 0.5*binsB[j]+0.5*binsB[j-1]

    for i in range(len(binsA)):
        for j in range(len(binsB)):
            dzAB[i, j] = (Za[i+1]-Za[i])*(Zb[j+1]-Zb[j])

    return Za, Zb, dzAB


def OuputPDF(ScalarA, ScalarB, nx, width, stride, binsA, binsB, dzAB,
             Z1M, Z2M, Z1Z1, Z2Z2, Z1Z2):

    ranges = [
        range(width, nx+width, stride),
        range(width, nx+width, stride),
        range(width, nx+width, stride),
    ]

    nbinsA = len(binsA)
    nbinsB = len(binsB)
    npdfs = np.prod([len(x) for x in ranges])
    pdfs = np.zeros((npdfs, 5 + (nbinsA-1)*(nbinsB-1)))

    print("Total PDFs are {0}".format(npdfs))

    # Loop on all the blocks
    for cnt, (i, j, k) in enumerate(itertools.product(ranges[0],
                                                      ranges[1], ranges[2])):

        block = np.s_[i-width:i+width, j-width:j+width, k-width:k+width]

        Ana_MeanA = np.mean(ScalarA[block])
        Ana_VarA  = np.mean((ScalarA[block] - Ana_MeanA)**2)
        Ana_MeanB = np.mean(ScalarB[block])
        Ana_VarB  = np.mean((ScalarB[block] - Ana_MeanB)**2)
        Ana_CovAB = np.mean((ScalarA[block] - Ana_MeanA)*(ScalarB[block] - Ana_MeanB))

        # Compute PDF
        # pdf, _, _, _ = stats.binned_statistic_2d(
        #     np.ravel(ScalarA[block]),
        #     np.ravel(ScalarB[block]),
        #     0,
        #     statistic="count",
        #     bins=[binsA, binsB],
        # )
        # pdf /= max(1e-16, np.sum(pdf))
        pdf, _, _ = np.histogram2d(np.ravel(ScalarA[block]),
                                   np.ravel(ScalarB[block]),
                                   bins=[binsA, binsB],
                                   density=True)
        PDF = np.reshape(pdf.flatten(order="F"), (len(binsA)-1,
                                                  len(binsB)-1), order="F")
        PDF = np.multiply(PDF, dzAB)

        MeanA = np.sum(np.multiply(PDF, Z1M))
        VarA = np.sum(np.multiply(PDF, Z1Z1)) - MeanA*MeanA
        MeanB = np.sum(np.multiply(PDF, Z2M))
        VarB = np.sum(np.multiply(PDF, Z2Z2)) - MeanB*MeanB
        CovAB = np.sum(np.multiply(PDF, Z1Z2)) - MeanA*MeanB

        pdfs[cnt, :5] = [MeanA, VarA, MeanB, VarB, CovAB]
        pdfs[cnt, 5:] = pdf.flatten(order="F")

    return pdfs


if __name__ == '__main__':

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='A tool to generate PDFs for training')
    parser.add_argument('-f', '--folder',
                        dest='folder',
                        help='Folder containing ensight files',
                        type=str,
                        default='.')

    parser.add_argument('-rNum', '--runNumber',
                        dest='rNumber',
                        help='HIT run number',
                        type=int,
                        default=10)
    parser.add_argument('-n',
                        '--nscalars',
                        dest='nscalars',
                        help='Number of scalar configurations to be used',
                        type=int,
                        default=1)
    parser.add_argument('-w',
                        '--width',
                        dest='width',
                        help='Width of the box for filtering',
                        type=int,
                        default=32)
    parser.add_argument('-s',
                        '--stride',
                        dest='stride',
                        help='Number of cells to skip',
                        type=int,
                        default=8)
    parser.add_argument('-b',
                        '--bins',
                        dest='bins',
                        help='Number of bins for the PDF',
                        type=int,
                        default=64)

    parser.add_argument('-ts',
                        '--tstart',
                        dest='tstart',
                        help='Starting time step of ensight data',
                        type=int,
                        default=17)

    parser.add_argument('-te',
                        '--tend',
                        dest='tend',
                        help='Ending time step of ensight data',
                        type=int,
                        default=18)

    parser.add_argument('-o',
                        '--output',
                        dest='output',
                        help='File name to dump the pdfs for training',
                        type=str,
                        default='OutputPDF.csv')

    args = parser.parse_args()
    fdir = os.path.abspath(args.folder)

    scalarConfig = ['I1', 'I4', 'I5', 'L1', 'PI1', 'PI4', 'PI5', 'PL1']

    binsA, binsB, dzAB = genNonUniformBins(args.bins, args.bins, 0.1, True)
    Z1M, Z2M, Z1Z1, Z2Z2, Z1Z2 = getBinCenter(args.bins, args.bins, 0.1, True)

    for nconfig in range(args.nscalars):
        scFolderA = os.path.join(fdir, ('ZA' + scalarConfig[nconfig] +
                                        str(args.rNumber)))
        scFolderB = os.path.join(fdir, ('ZB' + scalarConfig[nconfig] +
                                        str(args.rNumber)))
        Allpdfs = np.empty([0, 5 + (args.bins)*(args.bins)])
        print("Working on scalar configuration {0}".format(
            scalarConfig[nconfig]))
        for it in range(args.tstart, args.tend):
            filenameA = os.path.join(scFolderA,
                                     ('ZA' + scalarConfig[nconfig] +
                                      str(args.rNumber)
                                      + '.' + str(it).zfill(6)))
            filenameB = os.path.join(scFolderB,
                                     ('ZB' + scalarConfig[nconfig] +
                                      str(args.rNumber)
                                      + '.' + str(it).zfill(6)))
            scA, scB, scC = ReadEnsight(filenameA, filenameB,
                                        256, args.width, 8)

            # Z1 -> A, Z2 -> B
            pdf = OuputPDF(scA, scB, 256, args.width, args.stride,
                           binsA, binsB, dzAB, Z1M, Z2M, Z1Z1, Z2Z2, Z1Z2)
            Allpdfs = np.vstack((Allpdfs, pdf))

        Allpdfs = pd.DataFrame(Allpdfs,
                               columns=["MeanA", "VarA",
                                        "MeanB", "VarB", "CovAB"] +
                               ["Y{0:04d}".format(i)
                                for i in range((args.bins)*(args.bins))],)
        Allpdfs.to_pickle(args.output+scalarConfig[nconfig]+".gz",
                          compression='infer')
        print("Wrote file {0}".format(args.output+scalarConfig[nconfig]+".gz"))

    # output timer
    end = time.time() - start
    print("Elapsed time " + str(timedelta(seconds=end)) +
          " (or {0:f} seconds)".format(end))


"""

 I am typically running this with stride = 32 and width = 64 and
 64x64 bin size. Since I have only one time step data at step = 17
 and one scalar configuration of I1
  I am with the following command
 %run EnsightPDFml.py -f ../../ensight-3D/ -n 1 -w 64 -s 32 -b 65 -ts 17 -te 18 -o I1_W64_S8_32768

 and so to include all scalar config, modify -n 1 to -n 11
 and include all time steps modify -ts (starting time index)
 and -te (ending time index)

 Also can you keep track of the total number of PDFs this generates

"""
