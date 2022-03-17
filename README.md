# Pressure_Profile

Pressure profile of halos

```
from pressure import pressure
pressure_profile = pressure(radius, mass, z, mass_profile=mass_profile, density_gas_profile=density_gas_profile)
```

radius = radius at which pressure is calculated 
mass = mass M200m of halos (can be array for mass history)
z = redshift of halos (can be array)
mass_profile = total mass enclosed at radius (same size as radius), if not given, NFW will be assumed
density_gas_profile = gas density at radius (same size as radius), if not given, KS01 will be assumed

Need colossus, numpy, scipy to work.
