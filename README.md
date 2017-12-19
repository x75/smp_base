

# Sensorimotor learning basic lib

A python library of code used for sensorimotor learning experiments
with simulations, robots, etc. The basic idea is to generate states
(sensors) with actions (motors), learn different predictive models
from that data to approximate the sensorimotor dynamics and then use
the models to infer future actions.

This repository is in an early stages of release which I push on the
occasion of sharing the smp\_sphero code <sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>. Still performing
incremental clean-up and refactoring plus additions from existing
other repositories so a lot of things might still change.


## Dependencies

The main dependencies are the standard ones such as numpy, scipy,
matplotlib, and sklearn which are needed regardless. You can either
install them via package manager

    apt-get install python-numpy python-scipy python-matplotlib python-sklearn

Optional modules are rlspy (recursive least squares implementation)
from <sup><a id="fnr.2" class="footref" href="#fn.2">2</a></sup> and jpype <sup><a id="fnr.3" class="footref" href="#fn.3">3</a></sup>, a java to python bridge which we use for
computing information theoretic measures with the java information
dynamics toolkit <sup><a id="fnr.4" class="footref" href="#fn.4">4</a></sup>. Additional dependencies which might be made
optional in the future are pandas, ros, pyunicorn, mdp, Oger, and igmm <sup><a id="fnr.5" class="footref" href="#fn.5">5</a></sup>.

Additional dependencies which might be made optional in the future are
*pandas*, *ros*, *pyunicorn*, *mdp*, *Oger*, *pyemd*, *IncSfa*.

    apt-get install python-pandas python-mdp

For installing a basic ROS stack see the wiki at
<https://www.ros.org/wiki>, you need to have the package python-rospy.

Pyunicorn does recurrence analysis and can be obtained from
<https://github.com/pik-copan/pyunicorn> or via pip.

Oger is an extension for MDP and can be obtained from <http://reservoir-computing.org/installing_oger>.


## Configuration

The path to some of the libraries can be set in the config file. See config.py.dist for possible options.


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> <https://github.com/x75/smp_sphero>

<sup><a id="fn.2" href="#fnr.2">2</a></sup> <https://github.com/bluesquall/rlspy>

<sup><a id="fn.3" href="#fnr.3">3</a></sup> Either python-jpype via apt, or from pip either jpypex or JPype1

<sup><a id="fn.4" href="#fnr.4">4</a></sup> <https://github.com/jlizier/jidt>

<sup><a id="fn.5" href="#fnr.5">5</a></sup> <https://github.com/x75/igmm/tree/smp>
