---
layout: post
title:  "Adding tensor support to the toy AutoGrad"

date:   2025-08-24 01:00:00 +0000
categories: jekyll update
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
      inlineMath: [['$$','$$']]
    }
  });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Table of contents
* this unordered seed list will be replaced
{:toc}

# Post

In the previous [post](https://aschrein.github.io/jekyll/update/2025/08/24/ext_autograd_to_vectors.html) I've demonstrated how to extend a toy automatic differentiation system to handle vectors, instead of scalar values.
In this update I'll extend it to handle tensors. We first need to add the main Tensor class. The job of this class is to represent multi-dimensional arrays and provide the necessary operations for automatic differentiation. This is implemented using a simple linear array and a rule to access elements based on their multi-dimensional indices. The rule is quite trivial, we just compute the dot product of the indices with the strides to compute a flat index. The strides are computed as a simple multiplication of the dimensions:

```python
strides = [1]
for d in reversed(shape):
    strides.insert(0, strides[0] * d)

indices = [0, 1, 2...]

flat_index = dot(indices, strides)

```

After that everything is pretty much the same as handling the vectors.

Source Code:


```python
import math
import random

def make_array(dim):
    arr = []
    for _ in range(dim):
        arr.append(0.0)
    return arr

def dims_get_total(dims):
    total = 1
    for d in dims:
        total *= d
    return total

"""
New basic building block class for compute.
Basically a flat array with a rule for accessing elements.
We're using a basic rule of linear strides.
"""
class Tensor:
    def __init__(self, shape):
        self.shape   = shape
        self.data    = make_array(dims_get_total(shape))
        self.strides = []
        self._compute_strides()

    def _compute_strides(self):
        total = 1
        for d in reversed(self.shape):
            self.strides.insert(0, total)
            total *= d

    def _get_flat_index(self, indices):
        if len(indices) != len(self.shape):
            raise ValueError("Incorrect number of indices")
        index = 0
        for i, idx in enumerate(indices):
            if not (0 <= idx < self.shape[i]):
                raise IndexError("Index out of bounds")
            index += idx * self.strides[i]
        return index

    def __getitem__(self, indices):
        index = self._get_flat_index(indices)
        return self.data[index]

    def __setitem__(self, indices, value):
        index = self._get_flat_index(indices)
        self.data[index] = value

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            result = Tensor(self.shape)
            for i in range(len(self.data)):
                result.data[i] = self.data[i] + other
            return result
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type: {type(other)}")
        if self.shape != other.shape:
            raise ValueError(f"Shapes must be the same: {self.shape} vs {other.shape}")
        result = Tensor(self.shape)
        for i in range(len(self.data)):
            result.data[i] = self.data[i] + other.data[i]
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = Tensor(self.shape)
            for i in range(len(self.data)):
                result.data[i] = self.data[i] * other
            return result

        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type: {type(other)}")
        if self.shape != other.shape:
            raise ValueError(f"Shapes must be the same: {self.shape} vs {other.shape}")
        result = Tensor(self.shape)
        for i in range(len(self.data)):
            result.data[i] = self.data[i] * other.data[i]
        return result
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            result = Tensor(self.shape)
            for i in range(len(self.data)):
                result.data[i] = self.data[i] - other
            return result

        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type: {type(other)}")
        if self.shape != other.shape:
            raise ValueError(f"Shapes must be the same: {self.shape} vs {other.shape}")
        result = Tensor(self.shape)
        for i in range(len(self.data)):
            result.data[i] = self.data[i] - other.data[i]
        return result

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            result = Tensor(self.shape)
            for i in range(len(self.data)):
                result.data[i] = self.data[i] / other
            return result

        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type: {type(other)}")
        if self.shape != other.shape:
            raise ValueError(f"Shapes must be the same: {self.shape} vs {other.shape}")
        result = Tensor(self.shape)
        for i in range(len(self.data)):
            result.data[i] = self.data[i] / other.data[i]
        return result
    
    def abs(self):
        result = Tensor(self.shape)
        for i in range(len(self.data)):
            result.data[i] = abs(self.data[i])
        return result

def get_list_shape(lst):
    shape = []
    while isinstance(lst, list):
        shape.append(len(lst))
        lst = lst[0] if lst else []
    return shape

def linearize_list(lst):
    result = []
    def recurse(sublist):
        if isinstance(sublist, list):
            for item in sublist:
                recurse(item)
        else:
            result.append(sublist)
    recurse(lst)
    return result

def tensor_from_list(lst):
    shape       = get_list_shape(lst)
    tensor      = Tensor(shape)
    tensor.data = linearize_list(lst)
    return tensor

# Compute graph basic building block
class AutoGradNode:
    def __init__(self, shape):
        # scalar valued gradient accumulator for the final dL/dp
        self.shape = shape
        self.grad = Tensor(shape)
        # dependencies for causation sort
        self.dependencies = []

    def zero_grad(self):
        self.grad = Tensor(shape=self.shape)

    # Overload operators to build the computation graph
    def __add__(self, other): return Add(self, other)
    def __mul__(self, other): return Mul(self, other)
    def __sub__(self, other): return Sub(self, other)

    # Get a topologically sorted list of dependencies
    # starts from the leaf nodes and terminates at the root
    def get_topo_sorted_list_of_deps(self):
        visited = set()
        topo_order = []

        def dfs(node): # depth-first search
            if node in visited:
                return
            visited.add(node)
            for dep in node.dependencies:
                dfs(dep)
            topo_order.append(node)

        dfs(self)

        return topo_order

    def get_pretty_name(self): return self.__class__.__name__

    # Pretty print the computation graph in DOT format
    def pretty_print_dot_graph(self):
        topo_order = self.get_topo_sorted_list_of_deps()
        _str = ""
        _str += "digraph G {\n"
        for node in topo_order:
            _str += f"    {id(node)} [label=\"{node.get_pretty_name()}\"];\n"
            for dep in node.dependencies:
                _str += f"    {id(node)} -> {id(dep)};\n"
        _str += "}"
        return _str
    
    def backward(self):
        topo_order = self.get_topo_sorted_list_of_deps()

        for node in topo_order:
            node.zero_grad() # we don't want to accumulate gradients

        for i in range(len(self.grad.data)):
            self.grad.data[i] = 1.0 # seed the gradient at the output node dL/dL=1

        # Reverse the topological order for backpropagation to start from the output
        for node in reversed(topo_order):
            # from the tip of the  graph down to leaf learnable parameters
            # Distribute gradients
            node._backward()

    # The job of this method is to propagate gradients backward through the network
    def _backward(self):
        assert False, "Not implemented in base class"

    # Materialize the numerical value at the node
    # i.e. Evaluate the computation graph
    def materialize(self):
        assert False, "Not implemented in base class"

# Any value that is not learnable
class Variable(AutoGradNode):
    def __init__(self, values, name=None):
        assert isinstance(values, Tensor), "Values must be a Tensor"
        super().__init__(shape=values.shape)
        self.values = values
        self.name = name

    def get_pretty_name(self):
        if self.name:
            return f"Variable({self.name})"
        else:
            return f"Constant({[f'{round(v, 3)}' for v in self.values.data]})"

    def materialize(self): return self.values

    def _backward(self):
        pass

Constant = Variable

# Learnable parameter with initial random value 0..1
class LearnableParameter(AutoGradNode):
    def __init__(self, shape):
        super().__init__(shape=shape)
        self.values = Tensor(shape)
        for i in range(len(self.values.data)):
            self.values.data[i] = random.random()

    def get_pretty_name(self):
        return f"LearnableParameter({[round(v, 3) for v in self.values.data]})"

    def materialize(self): return self.values

    def _backward(self):
        pass

class Reduce(AutoGradNode):
    def __init__(self, a, op='+'):
        super().__init__(shape=[1,])
        self.a            = a
        self.dependencies = [a]
        self.op           = op
        assert op in ['+'], "Only sum reduction is supported"

    def materialize(self): return tensor_from_list([sum(self.a.materialize().data)])

    def _backward(self):
        self.a.grad = self.a.grad + self.grad.data[0] # broadcast the gradient

class Square(AutoGradNode):
    def __init__(self, a):
        super().__init__(shape=a.shape)
        self.a            = a
        self.dependencies = [a]

    def materialize(self):
        x = self.a.materialize()
        return x * x

    def _backward(self):
        self.a.grad = self.a.grad + self.grad * 2.0 * self.a.materialize()

class Sub(AutoGradNode):
    def __init__(self, a, b):
        assert a.shape == b.shape, "Incompatible tensor dimensions"
        super().__init__(shape=a.shape)
        self.a            = a
        self.b            = b
        self.dependencies = [a, b]

    def materialize(self):
        return self.a.materialize() - self.b.materialize()

    def _backward(self):
        self.a.grad = self.a.grad + self.grad
        self.b.grad = self.b.grad - self.grad

class Add(AutoGradNode):
    def __init__(self, a, b):
        assert a.shape == b.shape, "Incompatible tensor dimensions"
        super().__init__(shape=a.shape)
        self.a = a
        self.b = b
        self.dependencies = [a, b]

    def materialize(self):
        return self.a.materialize() + self.b.materialize()

    def _backward(self):
        self.a.grad = self.a.grad + self.grad
        self.b.grad = self.b.grad + self.grad

class Mul(AutoGradNode):
    def __init__(self, a, b):
        assert a.shape == b.shape, "Incompatible tensor dimensions"
        super().__init__(shape=a.shape)
        self.a = a
        self.b = b
        self.dependencies = [a, b]

    def materialize(self):
        return self.a.materialize() * self.b.materialize()

    def _backward(self):
        self.a.grad = self.a.grad + self.grad * self.b.materialize()
        self.b.grad = self.b.grad + self.grad * self.a.materialize()

a = LearnableParameter(shape=[3,])
b = LearnableParameter(shape=[3,])

for epoch in range(3000):

    x = Variable(tensor_from_list([random.random(), random.random(), random.random()]), name="x")
    z = Square(x) * a + b
    loss = Reduce(Square(z - (Square(x) * Constant(tensor_from_list([1.777, 1.333, 0.333])) + Constant(tensor_from_list([1.55, 0.0, -1.666]))))) # L2 loss to Ax^2+B

    print(f"Epoch {epoch}: loss = {loss.materialize()}; a = {a.materialize()}, b = {b.materialize()}")
    # Backward pass
    # Gradient reset happens internally in the backward pass
    loss.backward()

    # Update parameters
    learning_rate = 0.01333

    for node in [a, b]:
        # print(f"grad = {node.grad}")
        for i in range(len(node.grad.data)):
            node.values.data[i] -= learning_rate * node.grad.data[i]

with open(".tmp/graph.dot", "w") as f:
    f.write(loss.pretty_print_dot_graph())


```


<script src="https://utteranc.es/client.js"
        repo="aschrein/aschrein.github.io"
        issue-term="pathname"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>