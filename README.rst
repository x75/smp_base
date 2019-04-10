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

The first bit are the standard libraries such as numpy, scipy,
matplotlib, and sklearn which are needed regardless. You can either
install them via package manager

.. code:: example

    sudo apt install python3-numpy python3-scipy python3-matplotlib python3-sklearn

Ultimately optional but recommended modules are rlspy for a recursive
least squares implementation and jpype, a java to python bridge.

rlspy is used as a reference update rule for prediction learning in many
models. Get it from https://github.com/bluesquall/rlspy by cloning into
your filesystem and

.. code:: example

    cd rlspy
    sudo pip3 install .

Then set the RLSPY variable in ``smp_base/config.py``.

Get jpype with ``sudo apt install python3-jpype`` or
``sudo pip3 install jpype1``.

This is needed for computing information theoretic measures with the
Java Information Dynamics Toolkit available from
https://github.com/jlizier/jidt. Download the latest distribution zip
from there, unpack it and set the JARLOC variable in
``smp_base/config.py`` to point to the absolte path of infodynamics.jar

Another measure that is used is the earth mover's distance, currently
there's two different implementations used: emd and pyemd.

emd from https://github.com/garydoranjr/pyemd, that is git clone it,

.. code:: example

    cd pyemd
    sudo pip3 install .

should install package emd.

pyemd comes via pip by running

.. code:: example

    sudo pip3 install pyemd

Additional packages we depend on at various places are pandas, ROS,
pyunicorn, mdp, but they can be installed later. Pandas and MDP can be
had from the distro with

.. code:: example

    apt-get install python-pandas python-mdp

Oger is an extension for MDP and can be obtained from
http://reservoir-computing.org/installing_oger.

For installing a basic ROS stack see the wiki at
https://www.ros.org/wiki, you need to have the package python-rospy
(Recently I have been building a minimal py3 version of ROS from
source).

Pyunicorn is used for recurrence analysis and can be obtained from
source at https://github.com/pik-copan/pyunicorn or via pip.

otl: online temporal learning library by harold soh w/ gaussian process
reservoir readout, loca fork on https://github.com/x75/otl

pypr: gaussian mixture models with conditional inference, local fork on
https://github.com/x75/pypr

igmm: incremental gaussian mixture models,
https://github.com/yumilceh/igmm

IncSfa: incremental slow feature analysis, untested
https://github.com/varunrajk/IncSFA, https://github.com/Kaixhin/IncSFA

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
