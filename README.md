# Field inversion example

Scripts to solve simple 2D Poisson adjoint equation to understand how adjoint and field inversion works. Uses numpy and autograd library.

## File descriptors
* create_ground_truth.py
  * Run this to create pseudo experimental results. You would need to run this before running the other adjoint examples.
* adjoint_example_discrete_k.py
  * Inverts conductivity field with conductivity varied individually at every node.
* adjoint_exmaple.py
  * Inverts conductivity field with basis functions.

For more information check out the blog post at: https://joelcfd.com/inverting-pdes-with-adjoints/
