

# Sensorimotor learning basic lib

A python library of code I use for sensorimotor learning experiments
with simulations, robots, etc. The basic idea is to generate states
(sensors) with actions (motors), learn different predictive models
from that data to approximate the sensorimotor dynamics and use the
models to compute future actions.

This repository is in an early stages of release which I push on the
occasion of sharing the smp\_sphero code <sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>. Still performing
incremental clean-up and refactoring plus additions from existing
private repository.

New idea for structure:

-   models: fit/predict (offline), fitpredict (online)
-   sensorimotor loops
-   data and logging


## Dependencies

We have the standard main dependencies of numpy, scipy, matplotlib,
and sklearn which are needed regardless. You can either install them
via package manager

    apt-get install python-numpy python-scipy python-matplotlib python-sklearn

Optional modules are rlspy (recursive least squares implementation)
from <sup><a id="fnr.2" class="footref" href="#fn.2">2</a></sup> and jpype <sup><a id="fnr.3" class="footref" href="#fn.3">3</a></sup>, a java to python bridge which we use for
computing information theoretic measures with the java information
dynamics toolkit <sup><a id="fnr.4" class="footref" href="#fn.4">4</a></sup>.

Additional dependencies which might be made optional in the future are
pandas, ros, pyunicorn, and mdp/Oger.

    apt-get install python-pandas python-mdp

For installing a basic ROS stack see the wiki at
<https://www.ros.org/wiki>, you need to have the package python-rospy.

Pyunicorn does recurrence analysis and can be obtained from
<https://github.com/pik-copan/pyunicorn> or via pip.

Oger is an extension for MDP and can be obtained from <http://reservoir-computing.org/installing_oger>.


## Configuration

The path to certain libraries can be set in the config file. See config.py.dist for possible options.


## Reservoir lib

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">reservoirs.py</td>
<td class="org-left">contains Reservoir class, LearningRules class, a  few utility functions and a main method</td>
</tr>


<tr>
<td class="org-left">&#xa0;</td>
<td class="org-left">that demonstrates basic use of the class. It can definitely be simplified (WiP)</td>
</tr>


<tr>
<td class="org-left">learners.py</td>
<td class="org-left">this model embeds the underlying adaptive model into the sensorimotor context</td>
</tr>


<tr>
<td class="org-left">eligibility.py</td>
<td class="org-left">basic eligibility windows used in a variant of learning rules</td>
</tr>


<tr>
<td class="org-left">smp\\\_thread.py</td>
<td class="org-left">thread wrapper that provides constant dt run loop and asynchronous sensor callbacks</td>
</tr>
</tbody>
</table>

You could try and run 

    python reservoirs.py

or

    python reservoirs.py --help

to see possible options. Documentation and examples upcoming.


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> <https://github.com/x75/smp_sphero>

<sup><a id="fn.2" href="#fnr.2">2</a></sup> <https://github.com/bluesquall/rlspy>

<sup><a id="fn.3" href="#fnr.3">3</a></sup> Either python-jpype via apt, or from pip either jpypex or JPype1

<sup><a id="fn.4" href="#fnr.4">4</a></sup> <https://github.com/jlizier/jidt>
