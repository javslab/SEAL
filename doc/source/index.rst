SEAL API Documentation
======================

Experimenting is a crucial part of machine learning projects and we believe that structured experimentations bring more understandable and explainable results. Following our belief, we propose Seal, a python framework that tries to improve structure and method in machine learning experimentations, in addition, to provide an autoML backend support.

In this way, we want the user to focus on the experimentation design phase, by defining a set of consistent parameters such as a metric to optimize, metrics to track, and two types of splitting strategies. When all necessary elements indicated

Afterward, the user can solve his problem using an autoML integrated backend. It is also, possible to pass parameters for shaping the autoML backend search space.

Seal is also built around the concept of components. Components allow the user to:

* Attach reusable code that can be shared across experimentations
* To materialize supplementary test hypotheses, and attach them to an experiment

This way, data scientists can test different approaches without having to develop everything from scratch and without losing track of the previous experiments.

For now, we provide a collection of base components along with an ensemble of APIs that can be used to build custom ones.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   installation   

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Base

   seal

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Components

   seal.components

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Utils

   seal.utils