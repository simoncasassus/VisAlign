import sys
import numpy as np

include_path = '/home/simon/common/python/include/'
sys.path.append(include_path)

from VisAlign import Dask_editms

file_ms = '_p1_residuals.ms'
file_ms_output = '_p1_residuals_withPS.ms'

PA = 160.4
inc = 130

#F=0.5E-2
F=126E-6
F=1.185e-04
PSs=[]
x0=-216.18/1000.
y0=46.03/1000.
rho0=221.04/1000.
phi0=282.01
PSs.append({'x0':x0,'y0':y0,'F':F})
    
phi1=(phi0-180.)
x1=rho0*np.sin(phi1*np.pi/180.)
y1=rho0*np.cos(phi1*np.pi/180.)
PSs.append({'x0':x1,'y0':y1,'F':F})

phi2=phi1+2.*(PA-phi1)
x2=rho0*np.sin(phi2*np.pi/180.)
y2=rho0*np.cos(phi2*np.pi/180.)
PSs.append({'x0':x2,'y0':y2,'F':F})

phi3=phi2-180.
x3=rho0*np.sin(phi3*np.pi/180.)
y3=rho0*np.cos(phi3*np.pi/180.)
PSs.append({'x0':x3,'y0':y3,'F':F})

print("PSs",PSs)


Dask_editms.apply(file_ms,
                  file_ms_output=file_ms_output,
                  addPS=PSs)
