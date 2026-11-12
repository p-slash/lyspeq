from dataclasses import dataclass

import fitsio
import healpy
import numpy as np
from numba import njit


def safeInvert(fisher):
    di = np.diag_indices(fisher.shape[0])
    w = fisher[di] == 0
    fisher[di] = np.where(w, 1, fisher[di])
    invfisher = np.linalg.inv(fisher)
    invfisher[w, :] = 0
    invfisher[:, w] = 0
    fisher[di] = np.where(w, 0, fisher[di])
    return invfisher


@njit("f8[:, :](i8, f8[:])")
def _constructFisher(ndim, upper_fisher):
    fisher = np.empty((ndim, ndim))
    fari = 0
    for did in range(ndim):
        for jj in range(ndim - did):
            fisher[jj, jj + did] = upper_fisher[fari + jj]
            fisher[jj + did, jj] = upper_fisher[fari + jj]

        fari += ndim - did

    return fisher


@dataclass(slots=True)
class OneQsoPower:
    targetid: int
    zqso: float
    ra: float
    dec: float
    snr: float
    nqdim: int
    istart: int
    slicer: np.s_
    thetak: np.ndarray
    upper_compact_fisher: np.ndarray
    fisher: np.ndarray = None
    hpix: int = None

    @classmethod
    def fromImageHdu(cls, hdu: fitsio.hdu.IMAGE_HDU) -> "OneQsoPower":
        hdr = hdu.read_header()
        data = hdu.read()
        nqdim = hdr['NQDIM']
        istart = hdr['ISTART']
        thetak = (
            data[:nqdim]
            - data[nqdim:2 * nqdim]
            - data[2 * nqdim:3 * nqdim]
        )
        slicer = np.s_[istart:istart + nqdim]
    
        return cls(
            targetid=hdr['TARGETID'], zqso=hdr['ZQSO'],
            ra=hdr['RA'], dec=hdr['DEC'], snr=hdr['SNR'],
            nqdim=nqdim, istart=hdr['ISTART'], slicer=slicer,
            thetak=thetak, upper_compact_fisher=0.5 * data[3 * nqdim:].copy()
        )

    def __post_init__(self):
        self.fisher = _constructFisher(self.nqdim, self.upper_compact_fisher)


def makeListOneQsoPower(fnames: str | list[str]) -> list[OneQsoPower]:
    if isinstance(fnames, str):
        fnames = [fnames]

    list_of_qsos = []
    for f in fnames:
        if not f.endswith(".fits") and not f.endswith(".fits.gz"):
            print(f"File {f} does not have .fits extension")
            continue

        with fitsio.FITS(f) as fits:
            for hdu in fits:
                list_of_qsos.append(OneQsoPower.fromImageHdu(hdu))

    return list_of_qsos


@dataclass(slots=True)
class FullPowerEstimate():
    size: int
    thetak: np.ndarray  # actually thetak is F.thetak
    fisher: np.ndarray
    cov: np.ndarray = None  # cov is safeInvert(fisher)
    # Actual thetak: 0.5 * cov @ thetak

    @classmethod
    def fromListOneQsoPower(
            cls, list_of_qsos: list[OneQsoPower], size=None, fast=False
    ) -> "FullPowerEstimate":
        if size is None:
            size = max(qso.istart + qso.nqdim for qso in list_of_qsos)

        thetak = np.zeros(size)
        for qso in list_of_qsos:
            thetak[qso.slicer] += qso.thetak

        if not fast:
            fisher = np.zeros((size, size))
            for qso in list_of_qsos:
                fisher[qso.slicer, qso.slicer] += qso.fisher
        else:
            fisher = np.zeros((1, 1))

        return cls(size=size, thetak=thetak, fisher=fisher)

    def calculateCov(self):
        self.cov = safeInvert(self.fisher)


def makeHealpixSplitFromListOneQsoPower(
        list_of_qsos: list[OneQsoPower], nside: int, size=None
):
    if size is None:
        size = max(qso.istart + qso.nqdim for qso in list_of_qsos)

    unique_hpixels = set()
    dict_hpix_to_qsos = {}
    for qso in list_of_qsos:
        qso.hpix = healpy.ang2pix(nside, qso.ra, qso.dec, lonlat=True)
        unique_hpixels.add(qso.hpix)
        dict_hpix_to_qsos.setdefault(qso.hpix, []).append(qso)
    unique_hpixels = sorted(unique_hpixels)

    list_of_healpix_split = []
    for hpix in unique_hpixels:
        list_of_healpix_split.append(
            FullPowerEstimate.fromListOneQsoPower(
                dict_hpix_to_qsos[hpix], size=size
            )
        )

    return unique_hpixels, list_of_healpix_split


def getHealpixThetakAsMatrix(list_of_healpix_split: list[FullPowerEstimate], size=None):
    nhpix = len(list_of_healpix_split)
    if size is None:
        size = list_of_healpix_split[0].size
        assert all(fpe.size == size for fpe in list_of_healpix_split)

    thetak = np.zeros((nhpix, size))
    for i, est in enumerate(list_of_healpix_split):
        thetak[i, :est.size] = est.thetak

    return thetak


class Bootstrapper:
    """
    Class to perform bootstrap resampling of the quasar power estimates.
    Sample usage:
        files = glob.glob("output/desi-p1d-bootchunks-*.fits")
        booter = readBootchunks.Bootstrapper.fromFiles(files)

        cov_hpx = booter.bootstrapCovarianceHealpixFast(nside=32, nboot=1000)
    """
    def __init__(self, list_of_qsos: list[OneQsoPower]):
        self.list_of_qsos = list_of_qsos
        self.nqso = len(list_of_qsos)
        self.full_power_estimate = FullPowerEstimate.fromListOneQsoPower(list_of_qsos)
        self.full_power_estimate.calculateCov()
        self.size = self.full_power_estimate.size

    @classmethod
    def fromFiles(cls, fnames: str | list[str]) -> "Bootstrapper":
        list_of_qsos = makeListOneQsoPower(fnames)
        return cls(list_of_qsos)

    def bootstrapPerQuasar(self, nboot: int, seed=None, fast=True) -> list[FullPowerEstimate]:
        rng = np.random.default_rng(seed)
        list_of_bootstrap_estimates = []
        for _ in range(nboot):
            boot_indices = rng.integers(0, self.nqso, size=self.nqso)
            boot_qsos = [self.list_of_qsos[i] for i in boot_indices]
            list_of_bootstrap_estimates.append(
                FullPowerEstimate.fromListOneQsoPower(
                    boot_qsos, size=self.size, fast=fast
                )
            )
        return list_of_bootstrap_estimates

    def bootstrapCovarianceHealpixFast(
            self, nside: int, nboot: int, seed=None, batch_size=1000,
            return_full_estimates=False
    ) -> list[FullPowerEstimate]:
        """
        Compute the covariance of the power estimates using bootstrap resampling
        of the Healpix pixels. This is a fast method that does not require
        recomputing the full power estimate for each bootstrap sample.

        Weights are assigned to each Healpix pixel according to a Poisson
        distribution with mean 1, and the sum of the weights is normalized to
        the number of unique Healpix pixels.
        """
        unique_hpixels, list_of_healpix_split = makeHealpixSplitFromListOneQsoPower(
            self.list_of_qsos, nside, size=self.size
        )
        thetak = getHealpixThetakAsMatrix(list_of_healpix_split, size=self.size)
        nhpix = len(unique_hpixels)
        rng = np.random.default_rng(seed)
        bootstrap_estimates = np.empty((nboot, self.size))
        for jj in range(0, nboot, batch_size):
            jj2 = min(jj + batch_size, nboot)
            boot_weights = rng.poisson(1, size=(jj2 - jj, nhpix)).astype(float)
            factor = nhpix / boot_weights.sum(axis=1, keepdims=True)
            boot_weights *= factor
            np.matmul(boot_weights, thetak, out=bootstrap_estimates[jj:jj2])

        cov_y = np.cov(bootstrap_estimates, rowvar=False)
        C = self.full_power_estimate.cov
        if return_full_estimates:
            return 0.25 * C @ cov_y @ C, C.dot(bootstrap_estimates.T)
        return 0.25 * C @ cov_y @ C
