# Phase interpolation algorithm
This small repository contains phase interpolation examples based on the work from: 
- Chen, J., Zebker, H. A., & Knight, R. (2015). "A persistent scatterer interpolation
for retrieving accurate ground deformation over InSAR-decorrelated agricultural fields." 
Geophysical Research Letters, 42(21), 9294–9301. https://doi.org/10.1002/2015GL065031
- Wang, K., & Chen, J. (2022). "Accurate persistent scatterer identification based on
phase similarity of radar pixels".
IEEE Transactions on Geoscience and Remote Sensing, 1–1. https://doi.org/10.1109/TGRS.2022.3210868
- And implementations from: https://github.com/UT-Radar-Interferometry-Group

I set up two jupyter notebook examples: 
- simple_test.ipynb 
  Where i test the two algorithms using very simple arrays.
- ifgram_test.ipynb
  Example interpolations using actual interferograms located under the samples folder. 
  
The implementation is very basic and in need of improvement. 

- chen_interp function is my original implementation, which seems to work well, however is very slow. 
- chen_interp_v2 has changes suggested by Liang Yu (JPL). This works faster when interpolating small examples,
  but it kills the kernel when testing over more complex cases. I think the issue lies on how the sample,
  distances are being stored. It creates a growing array that probably becomes too large when interpolating 
  using a large sampling radius.

- Latest improvements added by Scott Staniewicz (JPL) under dev/interp_numba.py.
  Examples are shown in both test jupyter notebooks.
  Numba and pymp are required.
