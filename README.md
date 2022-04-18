# polytropic

This is a small code for solving linear perturbations of stars, which follows the code GYRE (https://gyre.readthedocs.io/en/stable/), but solved complex eigenvalues in a new way. See arXiv:2203.11809.

### What the code can do
- Make a polytropic star with rotation to the lowest order (as a function of only radius), a "rotational profile" (not exactly the same thing) must be given.
- Solve non-adiabatic non-radial modes of linear perturbations.
- Low-viscosity perturbations as shown in arXiv:2203.11809.
- Read mesa files.

### What the code can potentially do
- Implement adiabatic and/or radial mode calculations.
- Solve other similar linear eigenvalue problems.
- For science backgrounds, also see https://gyre.readthedocs.io/en/stable/.

### Caveats
- There are speed optimizations for matrix determinant calculation, but not for matrix construction. This can improve the speed significantly.
- Jump conditions are not included yet.

### The demo notebook
- The notebook can produce Figs. 1, 2 (with `low_viscosity=True`), 3, &4 (with `low_viscosity=False`) of arXiv:2203.11809.
- The total run time is ~20 min (Intel Core i7-1065G7) for each, due to lack of optimization.

### Citations, if it's useful
https://ui.adsabs.harvard.edu/abs/2022arXiv220311809S/abstract
```
@ARTICLE{ShiFuller2022,
       author = {{Shi}, Yanlong and {Fuller}, Jim},
        title = "{Viscous and centrifugal instabilities of massive stars}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Solar and Stellar Astrophysics},
         year = 2022,
        month = mar,
          eid = {arXiv:2203.11809},
        pages = {arXiv:2203.11809},
archivePrefix = {arXiv},
       eprint = {2203.11809},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220311809S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
