import numpy as np
import camb
import fitsio

zp = 2.4
karr_out = np.geomspace(1e-4, 2e2, 2000)

# Table 2 right-most column
ombh2 = 0.02242
omch2= 0.11933
mnu = 0.06
h = 0.6766
As = 2.105e-9
ns = 0.9665
tau = 0.0561

camb_params = camb.set_params(
    redshifts=sorted([zp], reverse=True),
    WantCls=False, WantScalars=False,
    WantTensors=False, WantVectors=False,
    WantDerivedParameters=False,
    WantTransfer=True, kmax=2e2,
    omch2=omch2,
    ombh2=ombh2,
    omk=0.,
    H0=100.0 * h,
    ns=ns,
    As=As,
    mnu=mnu,
    tau=tau
)
camb_results = camb.get_results(camb_params)
Om = (ombh2 + omch2) / h**2
Or = camb_results.get_Omega('photon')

camb_interp = camb_results.get_matter_power_interpolator(
    nonlinear=False, hubble_units=True, k_hunit=True)

lnk = np.log(karr_out)
lnP = np.log(camb_interp.P(zp, karr_out))

with fitsio.FITS(
        f"camb_linear_power_spectrum_{zp:.1f}.fits", 'rw', clobber=True
) as fts:
    hdr = {
        'zpivot': zp, 'Om': Om, 'Or': Or, 'hubble': h,
        'ombh2': ombh2, 'omch2': omch2, 'mnu': mnu,
        'As': As, 'ns': ns, 'tau': tau
    }
    fts.write([lnk, lnP], names=['LNK', 'LNP'], extname='PLINEAR', header=hdr)
