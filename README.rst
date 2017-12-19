===============================
Sensorimotor learning basic lib
===============================

    :Author: Oswald Berthold



1 Sensorimotor learning basic lib
---------------------------------

A python library of code used for sensorimotor learning experiments
with simulations, robots, etc. The basic idea is to generate states
(sensors) with actions (motors), learn different predictive models
from that data to approximate the sensorimotor dynamics and then use
the models to infer future actions.

This repository is in an early stages of release which I push on the
occasion of sharing the smp\_sphero code  [1]_ . Still performing
incremental clean-up and refactoring plus additions from existing
other repositories so a lot of things might still change.

1.1 Dependencies
~~~~~~~~~~~~~~~~

The main dependencies are the standard ones such as numpy, scipy,
matplotlib, and sklearn which are needed regardless. You can either
install them via package manager

::

    apt-get install python-numpy python-scipy python-matplotlib python-sklearn

Optional modules are rlspy (recursive least squares implementation)
from  [2]_  and jpype  [3]_ , a java to python bridge which we use for
computing information theoretic measures with the java information
dynamics toolkit  [4]_ . Additional dependencies which might be made
optional in the future are pandas, ros, pyunicorn, mdp, Oger, and igmm  [5]_ .

Additional dependencies which might be made optional in the future are
*pandas*, *ros*, *pyunicorn*, *mdp*, *Oger*, *pyemd*, *IncSfa*.

::

    apt-get install python-pandas python-mdp

For installing a basic ROS stack see the wiki at
`https://www.ros.org/wiki <https://www.ros.org/wiki>`_, you need to have the package python-rospy.

Pyunicorn does recurrence analysis and can be obtained from
`https://github.com/pik-copan/pyunicorn <https://github.com/pik-copan/pyunicorn>`_ or via pip.

Oger is an extension for MDP and can be obtained from `http://reservoir-computing.org/installing_oger <http://reservoir-computing.org/installing_oger>`_.

1.2 Configuration
~~~~~~~~~~~~~~~~~

The path to some of the libraries can be set in the config file. See config.py.dist for possible options.


.. [1] `https://github.com/x75/smp_sphero <https://github.com/x75/smp_sphero>`_

.. [2] `https://github.com/bluesquall/rlspy <https://github.com/bluesquall/rlspy>`_

.. [3] Either python-jpype via apt, or from pip either jpypex or JPype1

.. [4] `https://github.com/jlizier/jidt <https://github.com/jlizier/jidt>`_

.. [5] `https://github.com/x75/igmm/tree/smp <https://github.com/x75/igmm/tree/smp>`_
