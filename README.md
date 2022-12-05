# Phase interpolation algorithm
This small repository contains 2 version of the phase unwrapping algorithm under dev/phase_inter.py. 
The two implementations are based on Chen, J., Zebker, H. A., & Knight, R. (2015). 
"A persistent scatterer interpolation for retrieving accurate ground deformation over InSAR-decorrelated agricultural fields." 
Geophysical Research Letters, 42(21), 9294–9301. https://doi.org/10.1002/2015GL065031


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
