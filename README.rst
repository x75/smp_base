Sensorimotor primitives basic library
=====================================

This tries to be the library of basic functions for a set of
sensorimotor learning experiments with simulators and real robots in my
thesis project *Robotic self-exploration and acquisition of sensorimotor
primitives*. The basic idea there is to have states that combine sensors
with actions and motor signals and learn predictive models from incoming
data to infer future actions from the approximated sensorimotor
dynamics.

This repository is in an early stages of release which I push on the
occasion of sharing the smp\_sphero code  [1]_. Still performing
incremental clean-up and refactoring plus additions from existing other
repositories so a lot of things might still change.

Dependencies
------------

The main dependencies are the standard ones such as numpy, scipy,
matplotlib, and sklearn which are needed regardless. You can either
install them via package manager

.. code:: example

    apt-get install python-numpy python-scipy python-matplotlib python-sklearn

Ultimately optional but recommended modules are rlspy for a recursive
least squares implementation and jpype, a java to python bridge

Clone rlspy from https://github.com/bluesquall/rlspy somewhere into your
filesystem and

.. code:: example

    pip3 install .

from within the rlspy directory or set the RLSPY variable in
``smp_base/config.py``.

Get jpype with ``sudo apt install python3-jpype`` or
``sudo pip3 install jpype1``.

This is needed for computing information theoretic measures with the
Java Information Dynamics Toolkit available from
https://github.com/jlizier/jidt. Download the latest distribution zip
from there, unpack it and set the JARLOC variable in
``smp_base/config.py`` to point to the absolte path of infodynamics.jar

Additional packages we depend on at various places are pandas, ros,
pyunicorn, mdp, Oger, pyemd, IncSfa and igmm, but they can be installed
later. Pandas and MDP can be had from the distro with

.. code:: example

    apt-get install python-pandas python-mdp

For installing a basic ROS stack see the wiki at
https://www.ros.org/wiki, you need to have the package python-rospy
(Recently I have been building a minimal py3 version of ROS from
source).

Pyunicorn is used for recurrence analysis and can be obtained from
source at https://github.com/pik-copan/pyunicorn or via pip.

Oger is an extension for MDP and can be obtained from
http://reservoir-computing.org/installing_oger.

Configuration
-------------

In a freshly cloned repository a local configuration file has to be
created from a template. To do this

.. code:: example

    cp smp_base/config.py.dist smp_base/config.py

and then edit the file ``smp_base/config.py`` and set the JARLOC and
RLSPY variables to matching values. Use the absolute path so they can be
found from anywhere.

Development
-----------

.. todoList::

Footnotes
=========

.. [1]
   https://github.com/x75/smp_sphero
