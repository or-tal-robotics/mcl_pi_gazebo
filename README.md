# mcl_pi_simulation
Particles Intersection (PI) is a method to fuse multiple particle filter estimators, where the cross-dependencies of the observations is unknown.
This package contain a PI implembtation for fusion the particle filters that estimates the state of a robot in a flat space.
This package contain two main nodes:
* particle_filter.py - an implemantation of a particle filter for robot localization using [1].
* particlesintersection.py - an implemantation of a particle intersection for cooperative localization using [2].

## Dependencies
The following python packges are required:
* python 3.*
* numpy
* matplotlib
* sklearn
* sciPy
* teb_local_planner for move_base

## Runing
For a simulation with 3 robots and move_base controller use:
'''
> roslaunch mcl_pi_move_base.launch
'''

For a remote control (needs 3 controllers) use:
'''
> roslaunch mcl_pi.launch
'''

## References
[1] Thrun S, Burgard W, Fox D. Probabilistic robotics. MIT press; 2005 Aug 19.

[2] Tslil, Or, and Avishy Carmi. "Information fusion using particles intersection." 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018.
 
