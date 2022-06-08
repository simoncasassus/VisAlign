from pyralysis.io import DaskMS
import astropy.units as u
from pyralysis.transformers import Gridder, HermitianSymmetry, DirtyMapper
from pyralysis.io import FITS
from astropy.units import Quantity


def gridvis(file_ms,
            imsize=2048,
            hermitian_symmetry=False,
            dx=None,
            wantpsf=False,
            wantdirtymap=False):

    print("processing: ", file_ms)
    x = DaskMS(input_name=file_ms)
    dataset = x.read()
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
        dx = dx_theo / 10.
    else:
        print("sky image pixels: ", dx.to(u.arcsec))

    # du = (1/(imsize*dx)).to(u.lambdas, equivalencies=lambdas_equivalencies())

    print("imsize", imsize)

    gridder = Gridder(imsize=imsize,
                      cellsize=dx,
                      padding_factor=1.0,
                      hermitian_symmetry=hermitian_symmetry)

    dirty_mapper = DirtyMapper(input_data=dataset,
                               imsize=imsize,
                               padding_factor=1.0,
                               cellsize=dx,
                               stokes="I,Q",
                               hermitian_symmetry=False)

    dirty_images_natural = dirty_mapper.transform()
    gridded_visibilities_nat = dirty_mapper.uvgridded_visibilities.compute()
    gridded_weights_nat = dirty_mapper.uvgridded_weights.compute()

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
