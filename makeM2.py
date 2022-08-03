import numpy as np
import grunner as grun
import pprint
import os.path
import Loader
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timezone

# The scipy fitter
from scipy.optimize import curve_fit

# everything in iminuit is done through the Minuit object, so we import it
from iminuit import Minuit

# we also need a cost function to fit and import the LeastSquares function
from iminuit.cost import LeastSquares


def fillgrun():
    """
    Fills up the grun structure so that the grun.getdefault_filename() 
    will return the correct name for a given value of m2 and N
    """
    grun.tag = "vevscan"
    grun.data["evolverType"] = "PV2HBSplit23NoDiffuse"
    grun.data["restart"] = True
    grun.data["H"] = 0.000
    grun.data["finaltime"] = 100000

    # These values should be changed  later by setm2andN so that
    # getdefault_filename() works. They are arbitrary at this point
    grun.data["mass0"] = -4.8110
    grun.data["NX"] = 24

    grun.data["outputfiletag"] = grun.getdefault_filename()


fillgrun()


def getNvalues():
    """
    Returns an array containing the N used in the simulation
    """
    return [24, 32, 48, 64]


def getm2values():
    """
    Returns the critical m2 , that is m2c, and an array containing 
    the m2 used in the simulation

    Returns:
        m2c (real) : the critical value of m2
        m (array)  : list of values for the array m2
    """
    m20 = -4.813
    m2vals0 = m20*np.linspace(1./1.08, 1.08, 9, endpoint=True)

    m2c = -4.8110
    dx = m2c - m2vals0[-3]

    u = np.logspace(-4., 1., 6, base=2, endpoint=True)

    x = np.zeros(6)
    m = np.zeros(6)

    for i in range(0, len(u)):
        x[i] = dx * u[i]
        m[i] = m2c - x[i]
    return m2c, m


def setm2andN(m2, N):
    """
    Sets the value of m2 and N in the grun data structure so that
    grun.getdefault_filename() works correctly. This routine should
    be called before calling getdefault_filename() or related functions.
    """
    if m2:
        grun.data["mass0"] = m2
    if N:
        grun.data["NX"] = N


def getloader(filename, starttime=20000):
    """
    Returns an initialized loader using the filename produced.
    """
    fn = '../' + filename + '/' + filename + '.h5'
    try:
        file = open(fn, "r")
        print("Found filename ", fn)
    except:
        print("Unable to process filename ", fn)
        raise SystemExit(1)

    return Loader.Loader(fn, starttime=starttime)

# Returns the mean value of <M^2>


def compute_static_M2(loader):
    """
    Compute the 1/4 <Ma Ma> and its uncertainty via blocking.

    Args:
        loader (Loader) : an initialized loader object

    Returns:
        mean (real) : the mean values of  <MaMa>
        err (real)  : the error in the mean value
    """
    phi0 = loader.load("phi0")
    phi1 = loader.load("phi1")
    phi2 = loader.load("phi2")
    phi3 = loader.load("phi3")
    M2 = phi0*phi0 + phi1*phi1 + phi2*phi2 + phi3*phi3
    return Loader.blocking(M2, nBlock=20)

# Deprecated: use compute_static_M2


def average_M2(loader, **kwargs):
    """

    Compute the 1/4 <Ma Ma> and its uncertainty via blocking.

    Args:
        loader (Loader) : an initialized loader object
        **kwargs : additional options passed on to the loader object

    Returns:
        mean (real) : the mean values of  <MaMa>
        err (real)  : the error in the mean value
    """
    first = True
    keys = ["phi0", "phi1", "phi2", "phi3"]
    for key in keys:
        if first:
            data = loader.load(key, **kwargs)
            first = False
        else:
            data = np.append(data, loader.load(key, **kwargs))

    return Loader.blocking(data*data, nBlock=20)


def vev2vsN():
    """
    Loops through the possible values of N and computes <M2>.

    On output saves the filename  vevscan_Nxxx_m-0501265_h000000_c00500_M2.txt.

    The columns of these files contain m2, N, <M2>, and error of <M2>.

    Notes:
        The values of m2 and N are encoded by getm2values() and getNvalues()

    """
    m2c, m = getm2values()
    Ns = getNvalues()

    print(np.insert(m, 0, m2c))
    for m2i in m:
        print(m2i)
        setm2andN(m2i, None)
        file_handler = open(
            grun.getdefault_filename_Nchange() + "_M2.txt", "w")
        for N in Ns:
            setm2andN(m2i, N)
            grun.data["outputfiletag"] = grun.getdefault_filename()
            loader = getloader(grun.data["outputfiletag"], starttime=20000)

            if loader == None:
                print("Cant find loader")
                raise SystemExit(1)
            else:
                print(loader)

            value, err = compute_static_M2(loader)
            data = np.array([m2i, N, value, err]).reshape(1, 4)
            np.savetxt(file_handler, data)

        file_handler.close()

# Helper function for fitvev2


def nonlinfit(linv, Sigma2, SF2):
    return Sigma2 + 0.1560282993375*(SF2*SF2/Sigma2)*linv*linv + 0.677355*SF2*linv

# Helper function for fitvev2


def linfit(linv, Sigma2, SF2):
    return Sigma2 + 0.677355*SF2*linv

# Helper function for fitvev2


def plot(xmin, xmax, func, npoints=400):
    x = np.linspace(xmin, xmax, npoints)
    y = func(x)
    return (x, y)


def fitvev(m2):
    """
    Fits a set of values of <M2> with various values of N with the Hasenfratz form. 

    Two fits are done the linear fit and the non-linear fit

    Args:
        m2 (real) : the value of the mass that is being fit.

    Returns:
        a row of data with containing 

        mass, Sigma2 (linear), Sigma2/f^2 (linear), errors,  Sigma2 (nonlin), Sigma2/f^2 (nonlinear) , errors, ...
        see below

    Notes:
        Before calling this function the filename_M2.txt files need to have been created by
        vev2vsN()


    """
    setm2andN(m2, None)
    fname = grun.getdefault_filename_Nchange() + '_M2.txt'

    # Use only N=32,48,64 that is why we select 1:
    M2 = np.loadtxt(fname)
    Linv = 1./M2[1:, 1]  # 1/L
    S2 = M2[1:, 2]  # Sigma2
    dS2 = M2[1:, 3]
    print("L values are ", 1./Linv)

    # We didn't use this it was working at one point
    # popt, pcov = curve_fit(linfit, Linv, S2, p0=None, sigma = dS2)

    # Do the linear fit. The output is stored in  mlin fit structure
    c2 = LeastSquares(Linv, S2, dS2, linfit)
    mlin = Minuit(c2, Sigma2=0.1, SF2=1.)
    mlin.migrad()
    mlin.hesse()
    print("### Linear Fit ####")
    print(mlin.parameters, mlin.values, mlin.errors)

    # plot the output of the best fit
    x, y = plot(0., 1./16., lambda x: linfit(x, *mlin.values))
    fname = grun.getdefault_filename_Nchange() + '_fit1.txt'
    np.savetxt(fname, np.vstack((x, y)).T)

    # Do the nonlinear fit
    c1 = LeastSquares(Linv, S2, dS2, nonlinfit)
    mnl = Minuit(c1, Sigma2=mlin.values[0], SF2=mlin.values[1])
    mnl.migrad()
    mnl.hesse()
    print("### NonLinear Fit ####")
    print(mnl.parameters, mnl.values, mnl.errors)

    # plot the output of the best fit
    x, y = plot(0., 1./16., lambda x: nonlinfit(x, *mnl.values))
    fname = grun.getdefault_filename_Nchange() + '_fit2.txt'
    np.savetxt(fname, np.vstack((x, y)).T)

    # estimate the errors in sigma
    Sigma = np.sqrt(mnl.values[0])
    SigmaP = np.sqrt(mnl.values[0] + mnl.errors[0])
    SigmaM = np.sqrt(mnl.values[0] - mnl.errors[0])
    SigmaStat = abs(SigmaP-SigmaM)/2.

    # estimate the systematic error
    # SigmaLin differs from the true Sigma by 1/(f^2L)^2.
    # We estimate Sigma as the true one.
    SigmaLin = np.sqrt(mlin.values[0])

    dSigma = Sigma*pow(abs(SigmaLin/Sigma - 1.), 3./2.)

    print(Sigma, SigmaP, SigmaM, SigmaStat, dSigma)

    return np.asarray([m2, mlin.values[0], mlin.values[1], mlin.errors[0], mlin.errors[1], mnl.values[0], mnl.values[1], mnl.errors[0], mnl.errors[1], Sigma, SigmaStat, dSigma])


def makeallfits():
    """ Loops through all values of m2 and does a fit to the static <M2>.

    Outputs:
        A file vevscan_Nxxx_allfits.txt with the results of fitvev for each
        value of m2i

    Notes:
        Prior to calling this function the files vevscan_Nxxx_all
    m2c, m = getm2values()

    """

    fh = open(grun.tag + "_Nxxx_allfits.txt", "w")
    for m2i in m:
        data = fitvev(m2i)
        np.savetxt(fh, data.reshape(1, len(data)))


def process_fit_results():
    """ Reads the files tag_Nxxx_allfits.txt and tag_Nxxx_M2.txt and produces summary data files

    Outputs:
        M2dictionary : a dictionary containing a number of fields

        produces the files tag_Nxxx_M2_fits.txt which concats the files of this sort for further analysis
    """
    m2c, m2 = getm2values()
    M2dictionary = {}
    allfitsdata = np.loadtxt(grun.tag + "_Nxxx_allfits.txt")

    for i in range(0, len(m2)):
        m2i = m2[i]
        setm2andN(m2i, None)
        filename = grun.getdefault_filename_Nchange() + '_M2.txt'
        data = np.loadtxt(filename)

        file_out = open(grun.getdefault_filename_Nchange() +
                        '_M2_fits.txt', "w")
        for row in data:
            N = int(row[1])
            setm2andN(m2i, N)
            filenameN = grun.getdefault_filename()
            M2dictionary[filenameN] = {'mass2': m2i,
                                       'N': N, 'M2': row[2], 'dM2': row[3]}

            row_out = np.hstack((row, allfitsdata[i]))
            np.savetxt(file_out, row_out.reshape(1, len(row_out)))

        file_out.close()
    return M2dictionary

################################################################################


def makeplots():
    m2c, m = getm2values()
    Ns = getNvalues()

    M2dictionary = loadm2()

    N = 64
    sigma2 = getsigma2()
    print(m, sigma2)
    for i in range(0, len(sigma2)):
        m2i = m[i]
        setm2andN(m2i, N)
        filename = grun.getdefault_filename()
        filename_phi0123 = grun.getdefault_filename() + "_phi0123.txt"
        data = np.loadtxt(filename_phi0123)
        time = np.linspace(0, data.shape[0] - 1, data.shape[0])
        print(time[0:5], time.shape)
        print(M2dictionary[filename])
        M2 = M2dictionary[filename]['M2']
        nu = 0.737707
        beta = 0.38
        d = 3.
        plt.errorbar(time/(m2c - m2i)**(nu*1), data[:, 0]/M2, data[:, 2]/M2)

    plt.xlim((0, 1000))
    plt.show()


def _test():
    a = np.asarray([1, 2, 3, 5])
    b = np.asarray([2, 3, 4, 6])
    X = np.vstack((a, b))
    Y = X
    print(X, Y)
    Y = np.roll(Y, -1, axis=1)
    print(X, Y)
    Z = np.einsum('ij,ij->j', X[:, 0:3], Y[:, 0:3])
    print(Z)
    u = np.zeros(5)
    print(u)
    Y = np.roll(Y, -1, axis=1)
    print(Y)
    print(Y.shape[1])
    print(Y[:, :None-1])


def compute_correlator_MM(O1, nTMax=5000, nblocks=20, decim=1):
    """ Computes a correlation function of M_a M_a

    Args:
        O1 (np.array): an array containing the fields (phi0, phi1, phi2, phi3).
            The shape is (4,:)
        nTMax (int): the maximum time separation t1 - t2, default=5000
        nblocks (int): The number of blocks for the error estimate 

    Returns:
        MM (array) : an array size nTMax containing the correlation function of MaMa(t)
        dMM (np.array) : an array size nTMax containing the associated error
    """
    MM = np.zeros(nTMax)
    dMM = np.zeros(nTMax)

    O2 = O1
    for t in range(0, MM.size):
        if t > 0:
            array = np.einsum('ij,ij->j', O1[:, :-t*decim], O2[:, :-t*decim])
        else:
            array = np.einsum('ij,ij->j', O1[:, :], O2[:, :])

        MM[t], dMM[t] = Loader.blocking(array, nBlock=nblocks)
        O2 = np.roll(O2, -decim, axis=1)

    return (MM, dMM)

# Returns the value of <M(t)M(0)>


def write_correlator_MM(loader, basename, **kwargs):
    """
    Writes the value of <Ma(t) Ma(0)> to a file basename_phi0123.txt
    """
    phi0 = loader.load("phi0")
    phi1 = loader.load("phi1")
    phi2 = loader.load("phi2")
    phi3 = loader.load("phi3")
    O1 = np.vstack((phi0, phi1, phi2, phi3))
    MM, dMM = compute_correlator_MM(O1, nTMax=500, decim=100)

    sr = Loader.StatResult((MM, dMM))
    fname = basename + "_phi0123.txt"
    sr.save_to_txt(fname)


def makecorrelators():
    # vev2vsN()
    m2c, m = getm2values()
    Ns = getNvalues()

    print(Ns, m)

    for m2i in m:
        for N in Ns:
            setm2andN(m2i, N)
            filename = grun.getdefault_filename()
            loader = getloader(filename)
            print(loader)
            write_correlator_MM(loader, filename)


if __name__ == "__main__":
    process_fit_results()
    # makeplots()
    #print("Hello world")
