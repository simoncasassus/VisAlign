from pyralysis.io import DaskMS
import numpy as np
import astropy.units as u
from pyralysis.transformers import Gridder, HermitianSymmetry, DirtyMapper
from pyralysis.io import FITS
from astropy.units import Quantity


def completeuvplane_complex(V_half):
    V_half = np.squeeze(V_half)
    nu, nv = V_half.shape
    Nside = max(nu, nv)
    V_full = np.zeros((Nside, Nside)).astype(complex)
    V_full[:, int(Nside / 2):] = V_half[:, 0:int(Nside / 2)]
    Vmatrix = np.matrix(V_full)
    V_H = Vmatrix.getH()
    V_H = np.asarray(V_H)
    V_H = np.flip(V_H, 0)
    V_H = np.rot90(V_H)
    V_full[:, 0:int(Nside / 2)] = V_H[:, 0:int(Nside / 2)]
    #V_full_R = V_full.real
    #V_full_I = V_full.imag
    #V_full_amp = V_full_R**2 + V_full_I**2
    #print(V_full_amp.shape)
    #Vtools.View(V_full_amp)
    return V_full


def completeuvplane(V_half):
    V_half = np.squeeze(V_half)
    nu, nv = V_half.shape
    Nside = max(nu, nv)
    V_full = np.zeros((Nside, Nside))
    V_full[:, int(Nside / 2):] = V_half[:, 0:int(Nside / 2)]
    V_H = V_full.transpose()
    V_H = np.flip(V_H, 0)
    V_H = np.rot90(V_H)
    V_full[:, 0:int(Nside / 2)] = V_H[:, 0:int(Nside / 2)]
    return V_full


def gridvis(file_ms,
            imsize=2048,
            hermitian_symmetry=True,
            dx=None,
            wantpsf=False,
            wantdirtymap=False):

    print("processing: ", file_ms)
    x = DaskMS(input_name=file_ms)
    dataset = x.read(filter_flag_column=True, calculate_psf=True)
    #dataset.field.mean_ref_dir
    #dataset.psf[0].sigma

    if hermitian_symmetry:
        h_symmetry = HermitianSymmetry(input_data=dataset)
        h_symmetry.apply()

    dx_theo = Quantity(dataset.theo_resolution)
    dx_theo = dx_theo.to(u.arcsec)
    print("theoretical formula for finest angular scale  ", dx_theo)
    print("recommended  pixel size", dx_theo / 7.)

    if dx == None:
        print("using theoretical formula for pixel size")
        dx = dx_theo / 7.
    else:
        print("sky image pixels: ", dx.to(u.arcsec))

    # du = (1/(imsize*dx)).to(u.lambdas, equivalencies=lambdas_equivalencies())

    print("imsize", imsize)

    gridder = Gridder(imsize=imsize,
                      cellsize=dx,
                      padding_factor=1.0,
                      hermitian_symmetry=hermitian_symmetry)

    dirty_mapper = DirtyMapper(
        input_data=dataset,
        imsize=imsize,
        padding_factor=1.0,
        cellsize=dx,
        stokes="I",  # "I,Q"
        hermitian_symmetry=hermitian_symmetry)

    dirty_images_natural = dirty_mapper.transform()
    gridded_visibilities_nat = np.squeeze(
        dirty_mapper.uvgridded_visibilities.compute())
    gridded_weights_nat = np.squeeze(dirty_mapper.uvgridded_weights.compute())
    if hermitian_symmetry:
        gridded_visibilities_nat = completeuvplane_complex(
            gridded_visibilities_nat)
        gridded_weights_nat = completeuvplane(gridded_weights_nat)

    if wantdirtymap:
        #dirty_image_natural = dirty_images_natural[0].data[0].compute()
        fits_io = FITS()
        print("punching dirty map ", wantdirtymap)
        fits_io.write(dirty_images_natural[0].data, output_name=wantdirtymap)
    if wantpsf:
        #dirty_beam_natural = dirty_images_natural[1].data[0].compute()
        fits_io = FITS()
        print("punching dirty beam ", wantpsf)
        fits_io.write(dirty_images_natural[1].data, output_name=wantpsf)

    return dx, gridded_visibilities_nat, gridded_weights_nat
