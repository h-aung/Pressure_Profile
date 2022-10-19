# Pressure_Profile

Pressure profile of halos

```
from pressure import pressure
pressure_profile = pressure(radius, mass, r200m, z,  conc_model='diemer19', mass_def='vir', cosmo=None)
```

radius = radius at which pressure is calculated (kpc/h)
mass = mass M200m of halos (Mpc/h)
r200m = physical r200m (kpc/h)
z = redshift of halos 
conc_model = concentration - mass relation assumed. same name in colossus
mass_def = mass definition. Change mass and r200m accordingly.
cosmo = cosmology assumed. Planck15 for default

The output is thermal pressure in cgs unit. 
For breakdown, see the last function "pressure", which includes total pressure, fraction of non-thermal pressure (pnth/ptot) from which you can get thermal or non-thermal pressure.


Need colossus, numpy, scipy to work.
