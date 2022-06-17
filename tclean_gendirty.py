#casa --log2term --nogui -c tclean_gendirty.py
import os
import sys

print(sys.argv)

input_ms = sys.argv[1]
print("sourcems", input_ms)

cellsize = sys.argv[2]

nxin = int(sys.argv[3])
nyin = int(sys.argv[4])

imagename = sys.argv[5]

#imagename = "tclean_IB17"
#pix_size = '0.003arcsec'
#pix_num = 2048

os.system("rm -rf " + imagename + "*")
tclean(
    vis=input_ms,
    imagename=imagename,
    gridder='standard',
    niter=0,
    interactive=False,
    #savemodel='modelcolumn',
    specmode='mfs',
    cell=cellsize,
    imsize=[nxin,nyin],
    #robust=float(robustparam),
    #scales = [0,10,30,90],
    #cyclefactor=1,
    #deconvolver='multiscale',
    weighting='natural',
    datacolumn='data',
    nterms=1,
    psterm=False,
    #datacolumn='corrected',
    #restoringbeam="",
    #threshold="0.0uJy"
)

exportfits(imagename=imagename + ".image", fitsimage=imagename + ".image.fits")
exportfits(imagename=imagename + ".psf", fitsimage=imagename + ".psf.fits")

