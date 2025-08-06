.. raw:: html

   <!-- markdownlint-disable -->

NVIDIA PhysicsNeMo Examples
===========================

Introduction
------------

This repository provides sample applications demonstrating use of
specific Physics-ML model architectures that are easy to train and
deploy. These examples aim to show how such models can help solve real
world problems.

Introductory examples for learning key ideas
--------------------------------------------

+-----------------------------------+-----------------------------------+
| Use case                          | Concepts covered                  |
+===================================+===================================+
| `Darcy Flow <./cfd/darcy_fno/>`__ | Introductory example for learning |
|                                   | basics of data-driven models on   |
|                                   | Physics-ML datasets               |
+-----------------------------------+-----------------------------------+
| `Darcy Flow (Data +               | Data-driven training with         |
| Physics) <                        | physics-based constraints         |
| ./cfd/darcy_physics_informed/>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `Lid Driven Cavity                | Purely physics-driven (no         |
| Flow <./cfd/ldc_pinns/>`__        | external simulation/experimental  |
|                                   | data) training                    |
+-----------------------------------+-----------------------------------+
| `Vortex                           | Introductory example for learning |
| Sheddin                           | the basics of MeshGraphNets in    |
| g <./cfd/vortex_shedding_mgn/>`__ | PhysicsNeMo                       |
+-----------------------------------+-----------------------------------+
| `Medium-range global weather      | Introductory example on training  |
| forecast using                    | data-driven models for global     |
| FCN-AFNO <./weather/fcn_afno/>`__ | weather forecasting               |
|                                   | (auto-regressive model)           |
+-----------------------------------+-----------------------------------+
| `Lagrangian Fluid                 | Introductory example for          |
| Flow <./cfd/lagrangian_mgn/>`__   | data-driven training on           |
|                                   | Lagrangian meshes                 |
+-----------------------------------+-----------------------------------+
| `Stokes Flow (Physics Informed    | Data-driven training followed by  |
| Fi                                | physics-based fine-tuning         |
| ne-Tuning) <./cfd/stokes_mgn/>`__ |                                   |
+-----------------------------------+-----------------------------------+

Domain-specific examples
------------------------

The several examples inside PhysicsNeMo can be classified based on their
domains as below:

   **NOTE:** The below classification is not exhaustive by any means!
   One can classify single example into multiple domains and we
   encourage the users to review the entire list.

..

   **NOTE:** \* Indicates externally contributed examples.

CFD
~~~

+-----------------------+-----------------------+-----------------------+
| Use case              | Model                 | Transient             |
+=======================+=======================+=======================+
| `Vortex               | MeshGraphNet          | YES                   |
| Shedding <./cfd/vor   |                       |                       |
| tex_shedding_mgn/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Drag prediction -    | MeshGraphNet, UNet,   | NO                    |
| External              | DoMINO, FigConvNet    |                       |
| Aero <./cfd/exter     |                       |                       |
| nal_aerodynamics/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Navier-Stokes        | RNN                   | YES                   |
| Flow <./cfd/n         |                       |                       |
| avier_stokes_rnn/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Gray-Scott           | RNN                   | YES                   |
| System <./cf          |                       |                       |
| d/gray_scott_rnn/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Lagrangian Fluid     | MeshGraphNet          | YES                   |
| Flow <./cf            |                       |                       |
| d/lagrangian_mgn/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Darcy Flow using     | Nested-FNO            | NO                    |
| Nested-FNOs <./cfd/d  |                       |                       |
| arcy_nested_fnos/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Darcy Flow using     | Transolver            | NO                    |
| Transolver\* <./cfd/  | (Transformer-based)   |                       |
| darcy_transolver/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Darcy Flow (Data +   | FNO (branch) and MLP  | NO                    |
| Physics Driven) using | (trunk)               |                       |
| DeepONet              |                       |                       |
| a                     |                       |                       |
| pproach <./cfd/darcy_ |                       |                       |
| physics_informed/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Darcy Flow (Data +   | FNO                   | NO                    |
| Physics Driven) using |                       |                       |
| PINO approach         |                       |                       |
| (Numerical            |                       |                       |
| gra                   |                       |                       |
| dients) <./cfd/darcy_ |                       |                       |
| physics_informed/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Stokes Flow (Physics | MeshGraphNet and MLP  | NO                    |
| Informed              |                       |                       |
| Fine-Tuning) <        |                       |                       |
| ./cfd/stokes_mgn/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Lid Driven Cavity    | MLP                   | NO                    |
| Flow                  |                       |                       |
| <./cfd/ldc_pinns/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Magnetohydrodynamics | FNO                   | YES                   |
| using PINO (Data +    |                       |                       |
| Physics               |                       |                       |
| Driven)\*             |                       |                       |
|  <./cfd/mhd_pino/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Shallow Water        | FNO                   | YES                   |
| Equations using PINO  |                       |                       |
| (Data + Physics       |                       |                       |
| Driven)\* <./cfd/sw   |                       |                       |
| e_nonlinear_pino/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Shallow Water        | GraphCast             | YES                   |
| Equations using       |                       |                       |
| Distributed           |                       |                       |
| GNNs <./cfd/swe       |                       |                       |
| _distributed_gnn/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Vortex Shedding with | MeshGraphNet          | YES                   |
| Temporal              |                       |                       |
| Attentio              |                       |                       |
| n <./cfd/vortex_shedd |                       |                       |
| ing_mesh_reduced/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+
| `Data Center          | 3D UNet               | NO                    |
| Airflow <             |                       |                       |
| ./cfd/datacenter/>`__ |                       |                       |
+-----------------------+-----------------------+-----------------------+

Weather
~~~~~~~

+-----------------------------------+-----------------------------------+
| Use case                          | Model                             |
+===================================+===================================+
| `Medium-range global weather      | FCN-SFNO                          |
| forecast using                    |                                   |
| FCN-SFNO <https://git             |                                   |
| hub.com/NVIDIA/modulus-makani>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `Medium-range global weather      | GraphCast                         |
| forecast using                    |                                   |
| Gr                                |                                   |
| aphCast <./weather/graphcast/>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `Medium-range global weather      | FCN-AFNO                          |
| forecast using                    |                                   |
| FCN-AFNO <./weather/fcn_afno/>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `Medium-range and S2S global      | DLWP                              |
| weather forecast using            |                                   |
| DLWP <./weather/dlwp/>`__         |                                   |
+-----------------------------------+-----------------------------------+
| `Coupled Ocean-Atmosphere         | DLWP-HEALPix                      |
| Medium-range and S2S global       |                                   |
| weather forecast using            |                                   |
| DLWP-HEA                          |                                   |
| LPix <./weather/dlwp_healpix/>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `Medium-range and S2S global      | Pangu                             |
| weather forecast using            |                                   |
| Pa                                |                                   |
| ngu <./weather/pangu_weather/>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `Diagonistic (Precipitation)      | AFNO                              |
| model using                       |                                   |
| AFNO <./weather/diagnostic/>`__   |                                   |
+-----------------------------------+-----------------------------------+
| `Unified Recipe for training      | AFNO, FCN-SFNO, GraphCast         |
| several Global Weather            |                                   |
| Forecasting                       |                                   |
| mode                              |                                   |
| ls <./weather/unified_recipe/>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `Generative Correction Diffusion  | CorrDiff                          |
| Model for Km-scale Atmospheric    |                                   |
| Dow                               |                                   |
| nscaling <./weather/corrdiff/>`__ |                                   |
+-----------------------------------+-----------------------------------+
| `StormCast: Generative Diffusion  | StormCast                         |
| Model for Km-scale, Convection    |                                   |
| allowing Model                    |                                   |
| Em                                |                                   |
| ulation <./weather/stormcast/>`__ |                                   |
+-----------------------------------+-----------------------------------+

Generative
~~~~~~~~~~

+-------------------------------------+-------------------------------+
| Use case                            | Model                         |
+=====================================+===============================+
| `Fluid                              | flow_reconstruction_diffusion |
| Super-resolution\* <./cfd           |                               |
| /flow_reconstruction_diffusion/>`__ |                               |
+-------------------------------------+-------------------------------+

Healthcare
~~~~~~~~~~

+------------------------------------------------------+--------------+
| Use case                                             | Model        |
+======================================================+==============+
| `Cardiovascular                                      | MeshGraphNet |
| Simulations\* <./healthcare/bloodflow_1d_mgn/>`__    |              |
+------------------------------------------------------+--------------+
| `Brain Anomaly                                       | FNO          |
| Detection <./healthcare/brain_anomaly_detection/>`__ |              |
+------------------------------------------------------+--------------+

Additive Manufacturing
~~~~~~~~~~~~~~~~~~~~~~

+------------------------------------------------------+--------------+
| Use case                                             | Model        |
+======================================================+==============+
| `Metal Sintering                                     | MeshGraphNet |
| Simulatio                                            |              |
| n\* <./additive_manufacturing/sintering_physics/>`__ |              |
+------------------------------------------------------+--------------+

Molecular Dymanics
~~~~~~~~~~~~~~~~~~

+------------------------------------------------------+--------------+
| Use case                                             | Model        |
+======================================================+==============+
| `Force Prediciton for Lennard Jones                  | MeshGraphNet |
| system <./molecular_dynamics/lennard_jones/>`__      |              |
+------------------------------------------------------+--------------+

Additional examples
-------------------

In addition to the examples in this repo, more Physics-ML usecases and
examples can be referenced from the `PhysicsNeMo-Sym
examples <https://github.com/NVIDIA/physicsnemo-sym/blob/main/examples/README.md>`__.

NVIDIA support
--------------

In each of the example READMEs, we indicate the level of support that
will be provided. Some examples are under active development/improvement
and might involve rapid changes. For stable examples, please refer the
tagged versions.

Feedback / Contributions
------------------------

We’re posting these examples on GitHub to better support the community,
facilitate feedback, as well as collect and implement contributions
using `GitHub issues <https://github.com/NVIDIA/physicsnemo/issues>`__
and pull requests. We welcome all contributions!
