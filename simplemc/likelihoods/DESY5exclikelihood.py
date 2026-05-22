from simplemc.likelihoods.BaseLikelihood import BaseLikelihood
import numpy as np
import scipy.linalg as la
from scipy.interpolate import interp1d
from simplemc.setup_logger import cdir
import pandas as pd

class DESY5excLikelihood(BaseLikelihood):
    def __init__(self, name, values_filename, cov_filename, ninterp=150):
        """
        This module calculates likelihood for DESY5 datasets.
        Parameters
        ----------
        name: str
            name of the likelihood
        values_filename: str
            directory and name of the data file
        cov_filename: str
            directory and name of the covariance matrix file
        ninterp: int
        """
        # first read data file
        self.name_ = name
        BaseLikelihood.__init__(self, name)
        print("Loading", values_filename)
        da = pd.read_csv(values_filename)

        # Apply the filter for zHD > 0.01
        self.ww = (da['zHD'] > 0.01)  # Filter condition
        self.zcmb = da['zHD'][self.ww]  # Apply filter to zHD
        self.zhelio = da['zHEL'][self.ww]  # Apply filter to zHEL
        self.mag = da['MU'][self.ww]  # Apply filter to magnitude
        self.dmag = da['MUERR_FINAL'][self.ww]  # Apply filter to magnitude error

        self.N = 1635  # Set N to 1635
        self.zcmb = self.zcmb[:self.N]  # Use only the first 1635 data points
        self.zhelio = self.zhelio[:self.N]
        self.mag = self.mag[:self.N]
        self.dmag = self.dmag[:self.N]

        # Loading covariance matrix and cutting it to match the filtered data
        self.syscov = np.loadtxt(cov_filename, skiprows=1).reshape((1829, 1829))  # Full matrix (1829x1829)
        
        # Cut the covariance matrix to 1635x1635
        self.cov = self.syscov[:self.N, :self.N]  # Cut to match the 1635 data points

        # Add diagonal errors (dmag^2) to the covariance matrix
        self.cov[np.diag_indices_from(self.cov)] += self.dmag**2
        self.xdiag = 1 / self.cov.diagonal()  # Diagonal before marginalizing constant                
        
        self.cov += 3**2
        self.zmin = self.zcmb.min()
        self.zmax = self.zcmb.max()
        self.zmaxi = 1.1 ## we interpolate to 1.1 beyond that exact calc
        print("DESY5: zmin=%f zmax=%f N=%i" % (self.zcmb.min(), self.zcmb.max(), self.N))
        self.zinter = np.linspace(1e-3, self.zmaxi, ninterp)
        self.icov = la.inv(self.cov)

    def loglike(self):
        # we will interpolate distance
        dist = interp1d(self.zinter, [self.theory_.distance_modulus(z) for z in self.zinter],
                        kind='cubic', bounds_error=False)(self.zcmb)
        who = np.where(self.zcmb > self.zmaxi)
        dist[who] = np.array([self.theory_.distance_modulus(z) for z in self.zcmb.loc[who]])
        tvec = self.mag-dist

        # tvec = self.mag-np.array([self.theory_.distance_modulus(z) for z in self.zcmb])
        # print (tvec[:10])
        # first subtract a rought constant to stabilize marginaliztion of
        # intrinsic mag.
        tvec -= (tvec*self.xdiag).sum() / (self.xdiag.sum())
        # print(tvec[:10])
        chi2 = np.einsum('i,ij,j', tvec, self.icov, tvec)
        # print("chi2=",chi2)
        return -chi2/2
    
class DESY5exc(DESY5excLikelihood):
    """
    Likelihood to full DESY5 compilation.
    """
    def __init__(self):
        DESY5excLikelihood.__init__(self, "DESY5exc", cdir+"/data/DES-SN5YR_HD.csv",
                                      cdir+"/data/covsys_000.txt")
