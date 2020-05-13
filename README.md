# ns2div

ns2div is a GPU accelerated solver for Navier-Stokes equations with variable density based on 
Density variable Navier-Stokes code with GPU acceleration on rectangular structured meshes. The code is related to the paper:

> Monica Dessole, Fabio Marcuzzi "**[Fully iterative ILU preconditioning of the unsteady Navier–Stokes equations for GPGPU](https://www.sciencedirect.com/science/article/pii/S0898122118306345?via%3Dihub)**", Computers & Mathematics with Applications
Volume 77, Issue 4, 15 February 2019, Pages 907-92.

The numerical scheme implemented is based on the <a role="button" href="https://wikis.univ-lille.fr/painleve/ns2ddv">NS2DDV Matlab toolbox</a>. Strang splitting is used to decouple the transpost problem concerning the density the from the Navier-Stokes problem, describing the velocity and the pressure of the fluid. The former is treated with a Finite Volume scheme, in the latter a second-order Finite Element projection scheme is used to decouple the pressure and the components of the velocity field, each of them is obtained by solving s sparse linear system with GMRES with an LU type preconditioner which is updated with the Simplified ITALU procedure. 

## Running ns2div

In order to run a test one shoud execute the file `test_profile.py` with the inputs `nbseg_x` (number of mesh intervals in the x-direction), `nbseg_y` (number of mesh intervals in the y-direction), `BENCHMARK` (EXAC, RTIN or DROP). For example

```console
python test_profile.py 8 8 EXAC
```
run the exact test case on a \$8 \times 8\$ structured mesh.



![alt-text](OUTPUTS/RTIN3_Re1000_40x80_italu1_proj2.png)



## Manual setup





