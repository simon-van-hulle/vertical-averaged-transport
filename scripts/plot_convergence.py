#!/usr/bin/env python3

__author__ = "Simon Van Hulle"

"""
Adapted script from other project.
For this reason, the naming does not really add up.
I will not take the time to fix this, since the output works as expected.
"""

import argparse
import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import convergence as c
import particle_model.helpers as h

logger = h.easy_logger(__name__, logging.INFO)


def readErrors(fileName):

    logger.info(f"\nReading errors from {h.file_link(fileName)}")

    if os.path.exists(fileName):
        data = np.genfromtxt(fileName, comments='#').T

        dts = data[0]
        coeffs = data[1:]
        convg_list = [c.Convg(errors=data[i]) for i in range(1, len(data))]

        with open(fileName, 'r') as f:
            for line in f.readlines():
                if line.startswith('# dt'):
                    convgNames = list(line.strip('#').split())[1:]
                    break

    else:
        logger.info(f"\t[ERROR]: Coefficient file {fileName} not present.")
        logger.info(f"\t[ERROR]: Check the error directory!\n")
        return None, None, None

    for i, name in enumerate(convgNames):
        convg_list[i].name = name

    return dts, convg_list


# def readOrder_j_k(filename):
#     jLine = None
#     kLine = None

#     with open(filename) as f:
#         for line in f.readlines():
#             if line.startswith("# Order"):
#                 jLine = line
#             elif line.startswith("# Factor"):
#                 kLine = line

#     jVals = re.sub("[#\n\+-]", "", jLine).split()[2::2]
#     kVals = re.sub("[#\n\+-]", "", kLine).split()[2::2]

#     return map(float, jVals), map(float, kVals)

def plotOrder(dts, convg, args):
    """
    NOTE: This is a very delicate, first implementation. Very likely to break.
    But works for what I need right now.
    """
    nSteps = 100
    dtVals = np.linspace(dts.min(), dts.max(), nSteps)

    convg.order_convg(dts)
    if args.scale == 'linear':
        plt.plot(dtVals, c.error_f(dtVals, convg.j, convg.k), '--',
                 label=rf"E = {convg.k:.2f} * $(\Delta t)^{{{convg.j:.2f}}}$")
    elif args.scale == 'log':
        plt.loglog(dtVals, c.error_f(dtVals, convg.j, convg.k), '--',
                   label=rf"E = {convg.k:.2f} * $(\Delta t)^{{{convg.j:.2f}}}$")


def plotErrors(dts, convg_list, names, args):
    numPlots = 0

    plt.axvline(0, color='k', linewidth=2)
    plt.axhline(0, color='k', linewidth=2)

    for convg in convg_list:
        if convg.name in names:
            logger.info(f"\tPlotting {convg.name}")

            if (args.scale == 'linear'):
                plt.plot(dts, convg.errors, 'o',
                         markersize=3, label=convg.name)

            elif (args.scale == 'log'):
                plt.loglog(dts, convg.errors, 'o',
                           markersize=3, label=convg.name)

            plotOrder(dts, convg, args)
            
            numPlots += 1

                

    if numPlots > 0:
        ax = plt.gca()
        xlabel = rf"$\Delta t$"
        ylabel = "Error"

        if args.scale == 'log':
            xlabel = rf"$\log(\Delta t)$"
            ylabel = rf"$\log(Error)$"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axes.autoscale()
        plt.ylim([args.ymin, args.ymax])
        plt.legend()

    else:
        logger.info("\tNothing to plot")

    return numPlots


def plotFiles(args):
    files = args.files
    names = args.names
    allCoeffNames = []
    numPlots = 0

    for fileName in files:
        dts, convg_list = readErrors(fileName)
        # plotOrder(dts, convg_list, args)

        if type(dts) == type(None):
            break

        allCoeffNames += [convg.name for convg in convg_list]
        numPlots += plotErrors(dts, convg_list, names or allCoeffNames, args)

    if numPlots < len(names or [None]):
        logger.warning(f"Not all of your specified errors were found.")
        logger.warning(f"The following errors are included in the files:")
        for coeffName in allCoeffNames:
            logger.info(f"\t- {coeffName}")

    return numPlots


def parse_args():
    parser = argparse.ArgumentParser(description='Plot errors')

    # Positional Arguments
    parser.add_argument('files', nargs='+', metavar='files',
                        help='File names of errors')

    # Optional Arguments
    parser.add_argument('-n', '--names', nargs='+', metavar='',
                        help='Names of errors to plot')
    parser.add_argument('--tmin', metavar='', type=float, default=None,
                        help='Minimum time for plot')
    parser.add_argument('--ymin', metavar='', type=float, default=None,
                        help='Minimum value for the errors')
    parser.add_argument('--ymax', metavar='', type=float, default=None,
                        help='Maximum value for the errors')
    parser.add_argument('-s', '--scale', metavar='', type=str, default='linear',
                        choices={'linear', 'log'}, help='Scale for the axes')
    parser.add_argument('-o', '--outFile', metavar='',
                        help="File name for output image")

    args = parser.parse_args()
    return args


def plotTerminalInput():
    # Parse terminal arguments
    args = parse_args()

    # Plot the required errors from the required files
    numPlots = plotFiles(args)

    # Show plot if any errors present
    if numPlots > 0:
        if args.outFile:
            plt.savefig(args.outFile)

        plt.tight_layout()
        plt.show()

    logger.info("\n\nDone.\n")


if __name__ == '__main__':

    plotTerminalInput()
