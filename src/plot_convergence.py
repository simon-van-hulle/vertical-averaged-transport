#!/bin/python3

__author__ = "Simon Van Hulle"

"""
Adapted script from other project.
For this reason, the naming does not really add up.
I will not take the time to fix this, since the output works as expected.
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import helpers as h

logger = h.easy_logger(__name__, logging.INFO)


def readErrors(fileName):

    logger.info(f"\nReading errors from file {fileName}")

    if os.path.exists(fileName):
        data = np.genfromtxt(fileName, comments='#').T

        times = data[0]
        coeffs = data[1:]

        with open(fileName, 'r') as f:
            for line in f.readlines():
                if line.startswith('# dt'):
                    coeffNames = list(line.strip('#').split())[1:]
                    break

    else:
        logger.info(f"\t[ERROR]: Coefficient file {fileName} not present.")
        logger.info(f"\t[ERROR]: Check the error directory!\n")
        return None, None, None

    return times, coeffs, coeffNames


def plotErrors(time, coeffs, coeffNames, names, args):
    numPlots = 0
    tmin = args.tmin
    # Only use tmin correction if this is possible! Otherwise proceed.
    if tmin and np.max(time) > tmin:
        xmin = np.where(time > tmin)[0][0]
    else:
        xmin = 0

    time = time[xmin:]

    for name in names:
        if name in coeffNames:
            logger.info(f"\tPlotting {name}")

            if (args.scale == 'linear'):
                plt.plot(time, coeffs[coeffNames.index(name), xmin:], ':o',
                            label=name)

            elif (args.scale == 'log'):
                plt.loglog(time, coeffs[coeffNames.index(name), xmin:], ':o',
                           label=name)

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
    tmin = args.tmin
    allCoeffNames = []
    numPlots = 0

    for fileName in files:
        times, coeffs, coeffNames = readErrors(fileName)

        if type(times) == type(None):
            break

        allCoeffNames += coeffNames
        numPlots += plotErrors(times, coeffs, coeffNames, names or coeffNames,
                               args)

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
    parser.add_argument('--scale', metavar='', type=str, default='linear',
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
