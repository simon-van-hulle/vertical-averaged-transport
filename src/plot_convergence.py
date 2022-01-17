#!/bin/python3

__author__ = "Simon Van Hulle"

"""
Adapted script from other project.
For this reason, the naming does not really add up.
I will not take the time to fix this, since the output works as expected.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def log(*args, **kwargs):
    print(*args, **kwargs)


def readErrors(fileName):

    log(f"\nReading errors from file {fileName}")

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
        log(f"\t[ERROR]: Coefficient file {fileName} not present.")
        log(f"\t[ERROR]: Check the error directory!\n")
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
            log(f"\tPlotting {name}")
            plt.loglog(time, coeffs[coeffNames.index(name), xmin:], label=name)
            numPlots += 1

    if numPlots > 0:
        ax = plt.gca()
        ax.set_xlabel(r"$\log(dt)$")
        ax.set_ylabel(r"$\log(Error)$")
        ax.axes.autoscale()
        plt.ylim([args.ymin, args.ymax])
        plt.legend()

    else:
        log("\tNothing to plot")

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
        log(f"\n[WARNING]: Not all of your specified errors were found.")
        log(f"The following errors are included in the files:")
        for coeffName in allCoeffNames:
            log(f"\t- {coeffName}")

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

    log("\n\nDone.\n")


if __name__ == '__main__':
    plotTerminalInput()
