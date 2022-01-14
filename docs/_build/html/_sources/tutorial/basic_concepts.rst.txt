Basic concepts
===============

*"Why does FrEIA even exist? RealNVP can be implemented in \~100 lines of code!"*

That is correct, but the concept of INNs is more general:
For any computation graph, as long as each node in the graph is invertible, and
there are no loose ends, the entire computation is invertible. This is also
true if the operation nodes have multiple in- or outputs, e.g. concatenation
(*n* inputs, 1 output). So we need a framework that allows to **define an arbitrary computation graph,
consisiting of invertible operations.**

For example, consider wanting to implement some complicated new INN
architecture, with multiple in- and outputs, skip connections, a conditional part, ...:
|complicatedINN|

To allow efficient prototyping and experimentation with such architectures,
we need a framework that can perform the following tasks:

* As the inputs of operations depend on the outputs of others, we have to
  **infer the order of operations**, both for the forward and the inverse
  direction.
* The operators have to be initialized with the correct input-
  and output sizes in mind (e.g. required number of weights), i.e. we have to
  perform **shape inference** on the computation graph.
* During the computation, we have to **keep track of intermediate results**
  (edges in the graph) and store them until they are needed.
* We want to use **pytorch methods and tools**, such as ``.cuda()``,
  ``.state_dict()``, ``DataParallel()``, etc. on the entire computation graph,
  without worrying whether they work correctly or having to fix them.

Along with an interface to define INN computation graphs and invertible
operators within, these are the main tasks that ``FrEIA`` addresses.

.. |complicatedINN| image:: ../../inn_example_architecture.png
                            :scale: 60

