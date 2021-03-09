|Logo|

.. image:: https://github.com/VLL-HD/FrEIA/workflows/CI/badge.svg
   :alt: Build Status

This is the **Fr**\ amework for **E**\ asily **I**\ nvertible **A**\ rchitectures (**FrEIA**).

* Construct Invertible Neural Networks (INNs) from simple invertible building blocks.
* Quickly construct complex invertible computation graphs and INN topologies.
* Forward and inverse computation guaranteed to work automatically.
* Most common invertible transforms and operations are provided.
* Easily add your own invertible transforms.

.. contents:: Table of contents
   :backlinks: top
   :local:

Papers
--------------

Our following papers use FrEIA, with links to code given below.

**"Training Normalizing Flows with the Information Bottleneck for Competitive Generative Classification" (2020)**

* Paper: `arxiv.org/abs/2001.06448 <https://arxiv.org/abs/2001.06448>`_
* Code: `github.com/VLL-HD/exact_information_bottleneck <https://github.com/VLL-HD/exact_information_bottleneck>`_

**"Disentanglement by Nonlinear ICA with General Incompressible-flow Networks (GIN)" (2020)**

* Paper: `arxiv.org/abs/2001.04872 <https://arxiv.org/abs/2001.04872>`_
* Code: `github.com/VLL-HD/GIN <https://github.com/VLL-HD/GIN>`_

**"Guided Image Generation with Conditional Invertible Neural Networks" (2019)**

* Paper: `arxiv.org/abs/1907.02392 <https://arxiv.org/abs/1907.02392>`_
* Supplement: `drive.google.com/file/d/1_OoiIGhLeVJGaZFeBt0OWOq8ZCtiI7li <https://drive.google.com/file/d/1_OoiIGhLeVJGaZFeBt0OWOq8ZCtiI7li>`_
* Code: `github.com/VLL-HD/conditional_invertible_neural_networks <https://github.com/VLL-HD/conditional_invertible_neural_networks>`_

**"Analyzing inverse problems with invertible neural networks." (2018)**

* Paper: `arxiv.org/abs/1808.04730 <https://arxiv.org/abs/1808.04730>`_
* Code: `github.com/VLL-HD/analyzing_inverse_problems <https://github.com/VLL-HD/analyzing_inverse_problems>`_


Installation
--------------

FrEIA has the following dependencies:

+---------------------------+-------------------------------+
| **Package**               | **Version**                   |
+---------------------------+-------------------------------+
| Python                    | >= 3.7                        |
+---------------------------+-------------------------------+
| Pytorch                   | >= 1.0.0                      |
+---------------------------+-------------------------------+
| Numpy                     | >= 1.15.0                     |
+---------------------------+-------------------------------+
| Scipy                     | >= 1.5                        |
+---------------------------+-------------------------------+

Through pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: sh

   pip install git+https://github.com/VLL-HD/FrEIA.git

Manually
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For development:

.. code:: sh

   # first clone the repository
   git clone https://github.com/VLL-HD/FrEIA.git
   cd FrEIA
   # install the dependencies
   pip install -r requirements.txt
   # install in development mode, so that changes don't require a reinstall
   python setup.py develop


Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The full manual can be found at
https://vll-hd.github.io/FrEIA
including

* `Quickstart guide <https://vll-hd.github.io/FrEIA/_build/html/tutorial/quickstart.html>`_
* `Tutorial <https://vll-hd.github.io/FrEIA/_build/html/tutorial/tutorial.html>`_
* `Examples <https://vll-hd.github.io/FrEIA/_build/html/tutorial/examples.html>`_
* `API documentation <https://vll-hd.github.io/FrEIA/_build/html/index.html#package-documentation>`_

.. |Logo| image:: docs/freia_logo.png
