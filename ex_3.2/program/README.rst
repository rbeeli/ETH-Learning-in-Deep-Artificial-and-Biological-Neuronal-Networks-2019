**********************************
Tutorial on Variational Inference
**********************************

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

In tutorial 3.2, you will be implementing parts of the Bayes-by-backprop 
algorithm by completing missing parts in the module :mod:`lib.utils`. 
The theory to derive at the forumlas that need to be implemented were subject
to the exercise session as well as exercise 3.1.

Tutorial 3.2 -- Oct. 23nd, 2019
###############################

In the tutorial sessions on Oct. 16nd, 2019 and Oct. 23nd, 2019 you have seen 
how to derive the ELBO (Evidence lower bound). 
In this exercise, you are going to implement the ELBO for the simple case 
of factorized (independent) Gaussian weights in your Artificial Neural Networks.

Please refer to the :ref:`API <api-reference-label>` to get an overview of the 
coding structure.

You can see your code in action by learning a simple regression
task via the script :mod:`main`. 
Please study the command-line arguments on how to use the script.

.. code-block:: console

  $ python3 main.py --help

To visually certify the quality of the obtained predictions, you can
visualize the results of a 1D regression task via

.. code-block:: console

  $ python3 main.py
  
We recommend playing around with command-line options such as 
the number of weight samples (only for testing) ``--weight_samples`` and 
``--num_train_samples`` in combintation of turning Bayes-by-backprop on and off
i.e. ``--bbb`` (if implemented correctly).
  
**Please refer to the assignment sheet to obtain further information on how exercises have to be handed in and how they are graded.**

.. _Bayes-by-backprop: https://arxiv.org/abs/1505.05424
