===============================
Sensorimotor learning basic lib
===============================

    :Author: opt



1 Sensorimotor learning basic lib
---------------------------------

A python library of code I use for sensorimotor learning experiments
with simulations, robots, etc. Performing incremental clean-up and
pulling from existing private repository. The basis idea is to
generate states (sensors) with actions (motors), learn different
predictive models from that data to approximate the sensorimotor
dynamics and use the models to compute future actions.

This repository is in an early stages of release which I push on the
occasion of sharing the smp\_sphero code  [1]_ .

New idea for structure:

- models: fit/predict (offline), fitpredict (online)

- sensorimotor loops

- data and logging

1.1 Dependencies
~~~~~~~~~~~~~~~~

We have the standard main dependencies of numpy, scipy, matplotlib,
and sklearn which are needed regardless. You can either install them
via package manager

::

    apt-get install python-numpy python-scipy python-matplotlib python-sklearn

Optional modules are rlspy (recursive least squares implementation)
from  [2]_  and jpype  [3]_ , a java to python bridge which we use for
computing information theoretic measures with the java information
dynamics toolkit  [4]_ .

Additional dependencies which might be made optional in the future are
pandas, ros, pyunicorn, and mdp/Oger.

::

    apt-get install python-pandas python-mdp

For installing a basic ROS stack see the wiki at
`https://www.ros.org/wiki <https://www.ros.org/wiki>`_, you need to have the package python-rospy.

Pyunicorn does recurrence analysis and can be obtained from
`https://github.com/pik-copan/pyunicorn <https://github.com/pik-copan/pyunicorn>`_ or via pip.

Oger is an extension for MDP and can be obtained from `http://reservoir-computing.org/installing_oger <http://reservoir-computing.org/installing_oger>`_.

1.2 Reservoir lib
~~~~~~~~~~~~~~~~~

.. table::

    +------------------+-------------------------------------------------------------------------------------------+
    | reservoirs.py    | contains Reservoir class, LearningRules class, a  few utility functions and a main method |
    +------------------+-------------------------------------------------------------------------------------------+
    | \                | that demonstrates basic use of the class. It can definitely be simplified (WiP)           |
    +------------------+-------------------------------------------------------------------------------------------+
    | learners.py      | this model embeds the underlying adaptive model into the sensorimotor context             |
    +------------------+-------------------------------------------------------------------------------------------+
    | eligibility.py   | basic eligibility windows used in a variant of learning rules                             |
    +------------------+-------------------------------------------------------------------------------------------+
    | smp\\\_thread.py | thread wrapper that provides constant dt run loop and asynchronous sensor callbacks       |
    +------------------+-------------------------------------------------------------------------------------------+

You could try and run 

::

    python reservoirs.py

or

::

    python reservoirs.py --help

to see possible options. Documentation and examples upcoming.


.. [1] `https://github.com/x75/smp_sphero <https://github.com/x75/smp_sphero>`_

.. [2] `https://github.com/bluesquall/rlspy <https://github.com/bluesquall/rlspy>`_

.. [3] Either python-jpype via apt, or from pip either jpypex or JPype1

.. [4] `https://github.com/jlizier/jidt <https://github.com/jlizier/jidt>`_
