<h1> Citation Information </h1>
The provided code accompanies the following work:
@ARTICLE{lederer2021gaussian,
  author        = {{Tesfazgi}, Samuel and {Lederer}, Armin and {Kunz}, Johannes F. and {Ord{\'o}{\~n}ez-Conejo}, Alejandro J. and {Hirche}, Sandra},
  title         = "{Personalized Rehabilitation Robotics based on Online Learning Control}",
  journal       = {arXiv e-prints},
  year          = {2021},
  eid           = {arXiv:2110.00481},
  archivePrefix = {arXiv},
  eprint        = {2110.00481},
  primaryClass  = {cs.LG},
}

Please acknowledge the authors in any academic publication that have made use of this code or parts of it by referencing to the paper.

The code is tested using MatlabR2019a. The most recent version is available at: https://gitlab.lrz.de/online-GPs/LoG-GPs


Please send your feedbacks or questions to:
armin.lederer_at_tum.de


<h1> Code Structure </h1>
There is a slight change in Simulink between R2017a and the newer versions that requires small differences in the implementation behind the Simulink block interface. Therefore, we currently maintain implementations for both versions with the same functionalities. We recommend some kind of low pass filter directly behind the LoG-GP block to mitigate large changes in the predictions at the beginning of learning. Code is tested for SimulinkR2019a and might not work for older versions. Please run startup to add necessary paths and include the "Real Time Learning" toolbox in the Simulink Model Explorer. If the Real Time Learning toolbox is still not available, open the Model Explorer and follow the explanations to fix it as automatically suggested by Simulink.

<h3> Examples </h3>
<h4> Computed Torque Control </h4>
A 2 degree of freedom robotic manipulator controlled by computed torque control, which is augmented by a LoG-GP model. The debugging version of the LoG-GP with online hyperparameter optimization is employed in this example.

<h4> Feedback Linearization </h4>
A 2 degree of freedom robotic manipulator controlled by feedback linearization with a LoG-GP model. The clean version of the LoG-GP with hyperparameter optimization but without debugging outputs is used in this example.

<h3> Library </h3>
<ul> <li> RealTimeLearning.slx: Real Time Learning Library for the Simulink Model Explorer </li>
<li> slblocks.m: this function automatically adds the library to Model Explorer </li>
</ul>

<h3> Source Code </h3>
Simulink allows two modes of execution: interpreted execution and code generation. Currently, our implementation only supports interpreted execution.

<h4> Interpreted Execution </h4>
<ul> <li> LoGGP_IE_RPROP_DEBUG.m: source code for the debbuging block of LoG-GPs with online hyperparameter optimization </li>
<li> LoGGP_IE_RPROP_DEBUG.m: source code for the final block of LoG-GPs with online hyperparameter optimization </li>
</ul>
