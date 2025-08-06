Nested Fourier Neural Operater for Darcy Flow
=============================================

This example demonstrates how to set up a data-driven model for a 2D
Darcy flow using the Nested Fourier Neural Operator (FNO) architecture
inside of PhysicsNeMo. Training progress can be tracked through
`MLFlow <https://mlflow.org/docs/latest/index.html>`__. This case is
parallelised to run in multi-GPU settings.

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

Install the required dependencies by running below:

.. code:: bash

   pip install -r requirements.txt

Start with generating the dataset for training:

.. code:: bash

   python generate_nested_darcy.py

which will create the folder ``./data`` with ``out_of_sample.npy``,
``training_data.npy``, ``validation_data.npy``.

To train the model on a single GPU, run

.. code:: bash

   python train_nested_darcy.py +model=ref0
   python train_nested_darcy.py +model=ref1

For training a model on two GPUs, run

.. code:: bash

   mpirun -n 2 python train_nested_darcy.py +model=ref0
   mpirun -n 2 python train_nested_darcy.py +model=ref1

To evaluate the model use:

.. code:: bash

   python evaluate_nested_darcy.py

Progress can be monitored using MLFlow. Open a new terminal and navigate
to the training directory, then run:

.. code:: bash

   mlflow ui -p 2458

View progress in a browser at http://127.0.0.1:2458

If training on a remote machine, set up a ssh tunnel to the server with
``LocalForward 8080 your_remote_machine_addr:8080``. ssh to the server
via the specified port, in this case ``8080``, navigate to the training
directory and launch mlflow server

.. code:: bash

   mlflow server --host 0.0.0.0 --port 8080

On your local machine, open a browser and connect to ``localhost:8080``.

Additional Information
----------------------

References
----------

-  `Real-time high-resolution CO2 geological storage prediction using
   nested Fourier neural operators <https://arxiv.org/abs/2210.17051>`__
-  `Fourier Neural Operator for Parametric Partial Differential
   Equations <https://arxiv.org/abs/2010.08895>`__
