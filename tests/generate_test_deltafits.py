import numpy as np
import fitsio


def get_mean_reso(wave, reso, weight, dwave):
    total_weight = np.sum(weight)
    reso = np.dot(reso, weight) / total_weight
    lambda_eff = np.dot(wave, weight) / total_weight

    central_idx = reso.argmax()
    off_idx = np.array([-2, -1, 1, 2], dtype=int)
    ratios = reso[central_idx] / reso[central_idx + off_idx]
    ratios = np.log(ratios)
    w2 = ratios > 0
    norm = np.sum(w2)
    new_ratios = np.zeros_like(ratios)
    new_ratios[w2] = 1. / np.sqrt(ratios[w2])

    rms_in_pixel = np.abs(off_idx).dot(new_ratios) / np.sqrt(2.) / norm
    return rms_in_pixel * 3e5 * dwave / lambda_eff


z_qso = 2.4
dwave = 0.8
w1, w2 = 1050. * (1 + z_qso), 1180. * (1 + z_qso)
n = int((w2 - w1) / dwave) + 1
wave = w1 + np.arange(n) * dwave
delta = np.random.default_rng().normal(size=n)
ivar = np.random.default_rng().uniform(size=n)

ndiags = 11
r_A = 0.6
resomat = np.empty((n, ndiags))
dd = (np.arange(ndiags) - ndiags // 2) * dwave / r_A
dd = np.exp(-dd**2 / 2)
dd /= dd.sum()
resomat[:, :] = dd

print(wave)
print(resomat)


hdr_dict = {
    'LOS_ID': 1,
    'TARGETID': 1,
    'RA': 0.1,
    'DEC': 0.1,
    'Z': z_qso,
    'BLINDING': False,
    'WAVE_SOLUTION': "lin",
    'MEANSNR': 0.,
    'RSNR': 2.0,
    'DELTA_LAMBDA': dwave,
    'SMSCALE': 16.
}

hdr_dict['MEANSNR'] = np.mean(np.sqrt(ivar))

cols = [wave, delta, ivar, resomat.astype('f8')]
hdr_dict['MEANRESO'] = get_mean_reso(wave, resomat.T, ivar, dwave)
print(hdr_dict['MEANRESO'])

fitsio.write(
    "testdelta.fits",
    cols, names=['LAMBDA', 'DELTA', 'IVAR', 'RESOMAT'], header=hdr_dict,
    extname="1", clobber=True)
