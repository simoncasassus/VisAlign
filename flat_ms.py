from pyralysis.io import DaskMS
import numpy as np
import astropy.units as units
#from pyralysis.transformers import Gridder, HermitianSymmetry, DirtyMapper
#from astropy.units import Quantity


def plaw_divide(mss, alpha, ref_freq,data_column='DATA'):
    ##if isinstance(mss,str):
    ##    print("processing ", mss)
    ##    ds = DaskMS(input_name=mss, chunks={'row': 1E8, 'chan': 30})
    ##    mss = ds.read(filter_flag_column=True, calculate_psf=True)
    ##    print("done reading")
        
    for ms in mss.ms_list:
        Vin = ms.visibilities.dataset[data_column].data
        spw_id = ms.spw_id
        nchans = mss.spws.nchans[spw_id]
        chans = mss.spws.dataset[spw_id].CHAN_FREQ.data.squeeze(
            axis=0).compute() * units.Hz
        freqs = chans.value
        freqs_broadcast = freqs[np.newaxis, :, np.newaxis]
        Vout = Vin / (freqs_broadcast / ref_freq)**alpha
        ms.visibilities.dataset[data_column].data = Vout
    
    mss.compute()
    
