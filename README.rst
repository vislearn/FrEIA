|Logo|

.. image:: https://github.com/vislearn/FrEIA/workflows/CI/badge.svg
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

**"Generative Classifiers as a Basis for Trustworthy Image Classification" (CVPR 2021)**

* Paper: https://arxiv.org/abs/2007.15036
* Code: https://github.com/RayDeeA/ibinn_imagenet

**"Training Normalizing Flows with the Information Bottleneck for Competitive Generative Classification" (Neurips 2020)**

* Paper: `arxiv.org/abs/2001.06448 <https://arxiv.org/abs/2001.06448>`_
* Code: `github.com/vislearn/IB-INN <https://github.com/vislearn/IB-INN>`_

**"Disentanglement by Nonlinear ICA with General Incompressible-flow Networks (GIN)" (ICLR 2020)**

* Paper: `arxiv.org/abs/2001.04872 <https://arxiv.org/abs/2001.04872>`_
* Code: `github.com/vislearn/GIN <https://github.com/vislearn/GIN>`_

**"Guided Image Generation with Conditional Invertible Neural Networks" (2019)**

* Paper: `arxiv.org/abs/1907.02392 <https://arxiv.org/abs/1907.02392>`_
* Supplement: `drive.google.com/file/d/1_OoiIGhLeVJGaZFeBt0OWOq8ZCtiI7li <https://drive.google.com/file/d/1_OoiIGhLeVJGaZFeBt0OWOq8ZCtiI7li>`_
* Code: `github.com/vislearn/conditional_invertible_neural_networks <https://github.com/vislearn/conditional_invertible_neural_networks>`_

**"Analyzing inverse problems with invertible neural networks." (ICLR 2019)**

* Paper: `arxiv.org/abs/1808.04730 <https://arxiv.org/abs/1808.04730>`_
* Code: `github.com/vislearn/analyzing_inverse_problems <https://github.com/vislearn/analyzing_inverse_problems>`_


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

   pip install FrEIA

Manually
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For development:

.. code:: sh

   # first clone the repository
   git clone https://github.com/vislearn/FrEIA.git
   cd FrEIA
   # install the dependencies
   pip install -r requirements.txt
   # install in development mode, so that changes don't require a reinstall
   python setup.py develop


Documentation
-----------------

The full manual can be found at
https://vislearn.github.io/FrEIA
including

* `Quickstart guide <https://vislearn.github.io/FrEIA/_build/html/tutorial/quickstart.html>`_
* `Tutorial <https://vislearn.github.io/FrEIA/_build/html/tutorial/tutorial.html>`_
* `Examples <https://vislearn.github.io/FrEIA/_build/html/tutorial/examples.html>`_
* `API documentation <https://vislearn.github.io/FrEIA/_build/html/index.html#package-documentation>`_


How to cite this repository
-------------------------------

If you used this repository in your work, please cite it as below:

.. code-block:: 
   
   @software{freia,
     author = {Ardizzone, Lynton and Bungert, Till and Draxler, Felix and Köthe, Ullrich and Kruse, Jakob and Schmier, Robert and Sorrenson, Peter},
     title = {{Framework for Easily Invertible Architectures (FrEIA)}},
     year = {2018-2022},
     url = {https://github.com/vislearn/FrEIA}
   }

.. |Logo| image:: docs/freia_logo_invertible.svg
