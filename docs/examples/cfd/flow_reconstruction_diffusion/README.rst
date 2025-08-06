.. raw:: html

   <!-- markdownlint-disable -->

Diffusion-based-Fluid-Super-resolution
======================================

PyTorch implementation of

**A Physics-informed Diffusion Model for High-fidelity Flow Field
Reconstruction**

(Links to paper: Journal of Computational Physics \| arXiv)

.. raw:: html

   <div style style=”line-height: 25%” align="center">
   <h3>Sample 1</h3>
   <img src="https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/images/reconstruction_sample_01.gif">
   <h3>Sample 2</h3>
   <img src="https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/images/reconstruction_sample_02.gif">
   </div>

Overview
--------

Denoising Diffusion Probablistic Models (DDPM) are a strong tool for
data super-resolution and reconstruction. Unlike many other deep
learning models which require a pair of low-res and high-res data for
model training, DDPM is trained only on the high-res data. This feature
is especially beneficial to reconstructing high-fidelity CFD data from
low-fidelity reference, as it allows the model to be more independent of
the low-res data distributions and subsequently more adaptive to various
data patterns in different reconstruction tasks.

Datasets
--------

Datasets used for model training and sampling can be downloaded via the
following links.

-  High resolution data (ground truth for the super-resolution task)
   (link)

-  Low resolution data measured from random grid locations (input data
   for the super-resolution task) (link)

Prerequisites
-------------

Install the requirements using:

.. code:: bash

   pip install -r requirements.txt

Getting Started
---------------

Download the high res and low res data and save the data files to the
subdirectory
``physicsnemo/examples/cfd/flow_reconstruction_diffusion/Kolmogorov_2D_data/``.

-  Note: The directory from which the downloaded dataset files are
   loaded is specified in the configuration yaml files at
   ``physicsnemo/examples/cfd/flow_reconstruction_diffusion/conf/``. In
   the case when the default relative file location in a yaml file
   cannot be correctly recognized, please replace the relative location
   with the absolute location. For example, in the configuration file
   ``physicsnemo/examples/cfd/flow_reconstruction_diffusion/conf/config_dfsr_train.yaml``,
   Line 24, the value of the key ‘data’ can be changed to an absolute
   file directory of the dataset file, e.g.,
   ``/<directory of physicsnemo>/examples/cfd/flow_reconstruction_diffusion/Kolmogorov_2D_data/kf_2d_re1000_256_40seed.npy``

Step 1 - Model Training

In directory
``physicsnemo/examples/cfd/flow_reconstruction_diffusion/``, run:

(without physics-informed conditioning)

``python train.py --config-name=config_dfsr_train``

or

(with physics-informed conditioning)

``python train.py --config-name=config_dfsr_cond_train``

Step 2 - Super-resolution

In directory
``physicsnemo/examples/cfd/flow_reconstruction_diffusion/``, run:

(without physics-informed conditioning)

``python train.py --config-name=config_dfsr_generate``

or

(with physics-informed conditioning)

``python train.py --config-name=config_dfsr_cond_generate``

This implementation is based on / inspired by:

-  https://github.com/ermongroup/SDEdit (SDEdit: Guided Image Synthesis
   and Editing with Stochastic Differential Equations)
-  https://github.com/ermongroup/ddim (Denoising Diffusion Implicit
   Models)
