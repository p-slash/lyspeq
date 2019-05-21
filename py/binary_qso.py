import struct

class BinaryQSO:
    """
    Read and write in binary format for quadratic estimator.

    Parameters
    ----------
    fname : str
        Filename to read or write.
    rw : char
        Mode to open file. Can be either r or w.
    
    __init__(self, fname, rw)
        Opens a file buffer.

    Attributes
    ----------
    file : File
        File buffer. Closed after read or save methods.

    wave : float
        Wavelength array in Angstrom.
    flux : float
        Normalized flux.
    error : float
        Error on flux.
    N : int
        Length of these arrays.
    
    z_qso : float
        Emission redshift of the quasar.
    dv : float
        Pixel width of the spectrum.
    SPECRES : int
        Spectral resolution of the instrument.
    S2N : float
        Signal to noise ratio of the entire spectrum given by KODIAQ.
    RA : float
        RA in radians
    DECL : float
        DECL in radians

    Methods
    -------
    save(wave, flux, error, N, z_qso, DECL, RA, S2N, SPECRES, dv)
        Saves the given parameters in binary format. Does not hold them as attributes.

    read()
        Reads the file. Saves as attributes and returns:
        N : int
            Number of pixels
        z_qso : double
            Emission redshift of the quasar.
        DECL : double
        RA : double
            Declination and right ascension in radians
        SPECRES : int
            Dimensionless resolving power.
        S2N : double
            Signal to noise
        dv : double
            Pixel width in km/s
        low_ob_l : double 
            Lower observed wavelength
        upp_ob_l : double
            Upper observed wavelength
        low_re_l : double
            Lower rest frame wavelength
        upp_re_l : double
            Upper rest frame wavelength
        wave : Array of N doubles
        flux : Array of N doubles
        error : Array of N doubles

    """

    def __init__(self, fname, rw):
        self.file  = open(fname, mode=rw + 'b')

    def save(self, wave, flux, error, N, z_qso, DECL, RA, S2N, SPECRES, dv): 
        # Set up binary data
        low_ob_l = wave[0]
        upp_ob_l = wave[-1]

        low_re_l = low_ob_l / (1. + z_qso)
        upp_re_l = upp_ob_l / (1. + z_qso)

        hdr = struct.pack('idddidddddd', N, z_qso, DECL, RA, SPECRES, S2N, dv, \
                          low_ob_l, upp_ob_l, low_re_l, upp_re_l)
        wave_bin = struct.pack('d' * N, *wave)
        flux_bin = struct.pack('d' * N, *flux)
        nois_bin = struct.pack('d' * N, *error)

        # Save to file
        self.file.write(hdr)
        self.file.write(wave_bin)
        self.file.write(flux_bin)
        self.file.write(nois_bin)
        self.file.close()

    def read(self):
        header_fmt  = 'idddidddddd'
        header_size = struct.calcsize(header_fmt)

        d = self.file.read(header_size)

        self.N, self.z_qso, self.DECL, self.RA, \
        self.SPECRES, self.S2N, self.dv, \
        low_ob_l, upp_ob_l, low_re_l, upp_re_l  = struct.unpack(header_fmt, d)

        array_fmt  = 'd' * self.N
        array_size = struct.calcsize(array_fmt)

        d           = self.file.read(array_size)
        self.wave   = struct.unpack(array_fmt, d)
        d           = self.file.read(array_size)
        self.flux   = struct.unpack(array_fmt, d)
        d           = self.file.read(array_size)
        self.error  = struct.unpack(array_fmt, d)
        
        self.file.close()

        return  self.N, self.z_qso, self.DECL, self.RA, self.SPECRES, self.S2N, self.dv, \
                low_ob_l, upp_ob_l, low_re_l, upp_re_l, \
                self.wave, self.flux, self.error



