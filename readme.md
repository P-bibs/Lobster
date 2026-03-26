# 🦞 Lobster

*Lobster is forked from Scallop, see below for the original Scallop readme*
*The current version of lobster has limitations that are being addressed for a future release. See below for more details*

## Configuration

Lobster is configured via environment variables. Unless otherwise noted, environment variables are "flags" and merely need to be set or unset, their values are not read.

| Variable         | Valid Values                             | What It Does                                                                                                                                          |
| ---------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `STRATUM`        | Two comma-separated integers, e.g. `0,3` | **Required.** Defines the inclusive range of strata to run on the GPU. `STRATUM=2,5` offloads strata 2 through 5.                                     |
| `NO_CHECK`       | Any value (flag)                         | Skips CPU-side correctness verification of GPU results and lets the GPU path fully replace CPU execution for those strata.                            |
| `NO_OPTIMIZE`    | Any value (flag)                         | Disables the GPU-specific program optimization pass (`gpu_optimize`). Strata are executed as-is.                                                      |
| `PRINT_OPTIMIZE` | Any value (flag)                         | Prints strata before and after the optimization pass (leader thread only).                                                                            |
| `LOG`            | Any value (flag)                         | Master logging toggle for the C++ GPU library. Gates all `hINFO`, `hWARN`, `dINFO`, `dWARN` output.                                                   |
| `log_stratum`    | Any value (flag)                         | Enables verbose Rust-side logging of EDB/IDB contents and per-relation output after GPU execution. *(Note: lowercase.)*                               |
| `LOG_STRATUM`    | Any value (flag)                         | Enables verbose C++-side logging of stratum details, IDB state before/after execution, and output IDB construction.                                   |
| `LOG_FROG`       | Any value (flag)                         | Logs leapfrog arena allocator operations: leader/follower creation, memory region boundaries, and allocation sizes.                                   |
| `TRACE_FINE`     | Any value (flag)                         | Enables fine-grained tracing spans for individual dataflow evaluation steps (evaluate_stable, evaluate_recent, project).                              |
| `USE_FROG`       | Any value (flag)                         | Enables the leapfrog (ring-buffer) arena allocator instead of the default per-allocation CUDA allocator. More efficient for iterative fixpoint loops. |
| `ARENA_SIZE`     | Integer (gigabytes), e.g. `16`           | Sets the arena size in GB when `USE_FROG` is enabled. Defaults to 90% of total GPU memory. Only meaningful with `USE_FROG`.                           |
| OVERHEAD         | Positive integer, e.g. `8`               | Hash table overhead multiplier — controls how much extra space hash tables allocate relative to the data. Default: `8`.                               |
| `COMPACT`        | Any value (flag)                         | Enables a compact normalization strategy during sort+dedup.                                                                                           |


## Features

Supported provenances:
* `unit` (a.k.a, "discrete"/"boolean")
* `diffminmaxprob`
* `difftopkproofs` (with k=1)

Support operations
* `intersect`
* `union`
* `project`
* `join`
* `find`
* `overwrite_one`
* `product`
* `difference`

Supported types
* u32
* f32
* bool
* symbol (covers most string use-cases)

## Temporary limitations
* Template specializations are AOT compiled with code generation instead of jit compiled.
* Batching is unstable and may cause crashes.
* Stratum offload selection is unstable and you should manual select which strata to offload.

## Experiments
To run the experiments, you may need to modify the docker file to make sure your PyTorch version, docker CUDA toolkit version, and host CUDA driver version agree. Additionally, for some of the experiments you will need ~80GB of VRAM available. Additionally, note that on multi-GPU systems you must set the `CUDA_VISIBLE_DEVICES` environment variable to the GPU you want to use, and that even if you have exclusive access to one of the machine's GPUs, long-running jobs on other GPUs can negatively impact performance. This means to achieve peak performance, you should make sure all other GPUs on the machine are idle.

## Lobster Build Instructions

### First Time

Make sure this repository is
1. in your home folder
2. the directory is named `scallop-v2`
3. you have a terminal open in the `scallop-v2` directory

Build the docker container
```
docker build -t $(whoami)-cuda-dev-24-125 .
```
If successful, start the docker container
```
./docker-start.sh
```
Activate the  Python virtual environment (you only have to do this once)
```
source /root/venv/bin/activate
```
Configure the Lobster cmake build (you only have to do this once)
```
cd scallop-v2
cd core/src/gpu_runtime/libsclgpu && ./configure.sh && cd -
```
Build `scallopy` and `scli` with Lobster enabled
```
make develop-scallopy build-scli-dev
```

### Successive Times
Make sure you're in the container (you don't need to build it again).
```
./docker-start.sh
```
Change to the scallop-v2 directory inside the container
```
cd scallop-v2
```
Activate the Python virtual environment
```
source /root/venv/bin/activate
```
Build the code (this time in release mode, if desired)
```
make develop-scallopy-release build-scli
```
Run an experiment
```
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=... # your desired GPU number
./evaluation/ablation.sh
```

### Profiling and Benchmarking

An example of profiling with `nsight-compute` might look like
```
STRATUM=12,13 WANDB_MODE=disabled /usr/local/cuda/bin/ncu -o <name_of_output_file> --call-stack --kernel-name-base demangled -k regex:<name_of_kernel_to_profile> -s <number_of_initial_calls_to_skip> --set full python /pacman_maze/run.py --grid-x 8 --grid-y 8
```

---

# Scallop

| [Website](https://scallop-lang.github.io) | [Documentation](https://scallop-lang.github.io/doc) | [Playground](https://scallop.build) | [Download](https://github.com/scallop-lang/scallop/releases) | [Publications](https://www.scallop-lang.org/resources.html#section-3) |
|---|---|---|---|---|

<p align="center">
  <img width="240" height="240" src="res/icons/scallop-logo-ws-512.png" />
</p>

Scallop is a language based on DataLog that supports differentiable logical and relational reasoning.
Scallop program can be easily integrated in Python and even with a PyTorch learning module. You can also use it as another DataLog solver.
Internally, Scallop is built on a generalized [Provenance Semiring](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1022&context=db_research) framework.
It allows arbitrary semirings to be configured, supporting Scallop to perform discrete logical reasoning, probabilistic reasoning, and differentiable reasoning.

### Files

Each operation is implemented in an identically named file (e.g.
`join.h`/`join.cu`, `filter.h`/`filter.cu`). The table class is in
`table.h`/`table.cu`. The main entry point is `lib.cu`. Table normalization is
in `normalize.h`/`normalize.cu` and is called from `incorporate_delta()` in
`dataflow.cu`.

The bridge between the Scallop Rust code and the Lobster C++ code is in `core/src/gpu_runtime`.

## Example

Here is a simple probabilistic DataLog program that is written in Scallop:

```
// Knowledge base facts
rel is_a("giraffe", "mammal")
rel is_a("tiger", "mammal")
rel is_a("mammal", "animal")

// Knowledge base rules
rel name(a, b) :- name(a, c), is_a(c, b)

// Recognized from an image, maybe probabilistic
rel name = {
  0.3::(1, "giraffe"),
  0.7::(1, "tiger"),
  0.9::(2, "giraffe"),
  0.1::(2, "tiger"),
}

// Count the animals
rel num_animals(n) :- n = count(o: name(o, "animal"))
```

## How to use

### Prerequisite

Install `rust` with `nightly` channel set to default.

``` bash
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
$ rustup default nightly
```

### Download and Build

``` bash
$ git clone https://github.com/scallop-lang/scallop.git
$ cd scallop
```

The following three binaries are available. Scroll down for more ways
to use Scallop!

``` bash
$ make install-scli # Scallop Interpreter
$ make install-sclc # Scallop Compiler
$ make install-sclrepl # Scallop REPL
```

### Using Scallop Interpreter

Scallop interpreter (`scli`) interprets a scallop program (a file with extension `.scl`).
You can install `scli` to your system using

``` bash
$ make install-scli
```

Then since `scli` is in your system path, you can simply run

``` bash
$ scli examples/animal.scl
```

Note that by default we don't accept probabilistic input.
If your program is proabalistic and you want to obtain the resulting probabilities, do

``` bash
$ scli examples/digit_sum_prob.scl -p minmaxprob
```

Note that the `-p` argument allows you to specify a provenance semiring.
The `minmaxprob` is a simple provenance semiring that allows for probabilistic reasoning.

### Using Scallop REPL

Scallop REPL (`sclrepl`) is an interactive command line interface for you to try various ideas with Scallop.
You can install `sclrepl` to your system using

``` bash
$ cargo install --path etc/sclrepl
```

Then you can run `sclrepl`. You can type scallop commands like the following

``` bash
$ sclrepl
scl> rel edge = {(0, 1), (1, 2)}
scl> rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
scl> query path
path: {(0, 1), (0, 2), (1, 2)}
scl>
```

### Using `scallopy`

`scallopy` is the python binding for Scallop.
It provides an easy to use program construction/execution pipeline.
With `scallopy`, you can write code like this:

``` python
import scallopy

# Create new context (with unit provenance)
ctx = scallopy.ScallopContext()

# Construct the program
ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", [(0, 1), (1, 2)])
ctx.add_rule("path(a, c) = edge(a, c)")
ctx.add_rule("path(a, c) = edge(a, b), path(b, c)")

# Run the program
ctx.run()

# Inspect the result
print(list(ctx.relation("path"))) # [(0, 1), (0, 2), (1, 2)]
```

In addition, `scallopy` can be seamlessly integrated with PyTorch.
Here's how one can write the `mnist_sum_2` task with Scallop:

``` python
class MNISTSum2Net(nn.Module):
  def __init__(self, provenance="difftopkproofs", k):
    super(MNISTSum2Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(10)))
    self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
    self.scl_ctx.add_rule("sum_2(a + b) = digit_1(a), digit_2(b)")

    # The `sum_2` logical reasoning module
    self.sum_2 = self.scl_ctx.forward_function("sum_2", list(range(19)))

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    (a_imgs, b_imgs) = x

    # First recognize the two digits
    a_distrs = self.mnist_net(a_imgs)
    b_distrs = self.mnist_net(b_imgs)

    # Then execute the reasoning module; the result is a size 19 tensor
    return self.sum_2(digit_1=a_distrs, digit_2=b_distrs)
```

To install, please do the following (also specified [here](etc/scallopy/readme.md)):

Assume you are inside of the root `scallop` directory.
First, we need to create a virtual environment for Scallop to operate in.

``` bash
# Mac/Linux (venv, requirement: Python 3.8)
$ make py-venv # create a python virtual environment
$ source .env/bin/activate # if you are using fish, use .env/bin/activate.fish

# Linux (Conda)
$ conda create --name scallop-lab python=3.8 # change the name to whatever you want
$ conda activate scallop-lab
```

And let's install the core dependencies

``` bash
$ pip install maturin
```

With this, we can build our `scallopy` library

``` bash
$ make install-scallopy
```

If succeed, please run some examples just to confirm that `scallopy` is indeed installed successfully.
When doing so (and all of the above), please make sure that you are inside of the virtual environment or
conda environment.

``` bash
$ python etc/scallopy/examples/edge_path.py
```

### Scallop VSCode Plugin

To install VSCode plugin from source, you can do the following, after making sure that `npm` is installed on your system

``` bash
$ npm install -g vsce
$ make vscode-plugin
```

After this, a new `.vsix` plugin will appear in the `etc/vscode-scl` directory, named `scallop-x.x.x.vsix`.
Next, please hold `cmd + shift + p` in VSCode and type "Install from VSIX".
In the pop-up window, choose the `.vsix` plugin we just generated, and the plugin will be installed.

## Scallop Language

### Fact Declaration

You can declare a single fact using the following syntax.
In each line you define a single atom with every argument being constant.

```
rel digit(0, 1) // non-probabilitic
rel 0.3::digit(0, 1) // probabilistic
```

Alternatively, you can declare a set of facts using the following syntax.

```
rel digit = {
  0.4::(0, 1),
  0.3::(0, 2),
  0.1::(0, 3),
}
```

### Rule Declaration

You can declare rule using traditional datalog syntax:

```
rel path(a, b) :- edge(a, b)
rel path(a, c) :- path(a, b), edge(b, c)
```

Alternatively, you can use a syntax similar to logic programming:

```
rel path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))
```

### Probabilistic Rule

It is possible to declare a probabilistic rule

```
rel 0.3::path(a, b) = edge(a, b)
rel 0.5::path(b, c) = edge(c, b)
```

### Negation

Scallop supports stratified negation, with which you can write a rule like this:

```
scl> rel numbers(x) = x == 0 or (numbers(x - 1) and x <= 10)
scl> rel odd(1) = numbers(1)
scl> rel odd(x) = odd(x - 2), numbers(x)
scl> rel even(y) = numbers(y), ~odd(y)
scl> query even
even: {(0), (2), (4), (6), (8), (10)}
```

### Aggregation

We support the following aggregations `count`, `min`, `max`, `sum`, and `prod`.
For example, if you want to count the number of animals, you can write

```
scl> rel num_animals(n) :- n = count(o: name(o, "animal"))
scl> query num_animals
num_animals: {(2)}
```

Here `n` is the final count; `o` is the "key" variable that you want to count on;
`name(o, "animal")` is the sub-formula that can pose constraint on `o`.

Naturally, the arguments that are not key and appears in both the sub-formula and
outside of sub-formula will become a `group-by` variable.
The following example counts the number of objects (`n`) of each color (`c`):

```
scl> rel object_color = {(0, "blue"), (1, "green"), (2, "blue")}
scl> rel color_count(c, n) :- n = count(o: object_color(o, c))
scl> query color_count
color_count: {("blue", 2), ("green", 1)}
```

The results says there are two `"blue"` objects and one `"green"` object, as expected.

For the aggregation such as `min` and `max`, it is possible to get the `argmax` and
`argmin` at the same time.
Building up from the previous object-color example, the following rule can extract the
color that has the most number of objects:

```
scl> rel max_color(c) :- _ = max[c](n: color_count(c, n))
scl> query max_color
max_color: {("blue")}
```

Note that we have `max[c]` denoting that we want to get `c` as the argument for `max`.
Also, we use a wildcard `_` on the left hand side of the aggregation denoting that we
don't care about the aggregation result.
The final answer here is `"blue"` since there are 2 of them, which is greater than that
of color `"green"`.

Combining all of these, you can have a query containing group by and argument simultaneously.
The following example builds on a table containing student, their class, and their grade:

```
rel class_student_grade = {
  (0, "tom", 50),
  (0, "jerry", 70),
  (0, "alice", 60),
  (1, "bob", 80),
  (1, "sherry", 90),
  (1, "frank", 30),
}

rel class_top_student(c, s) = _ = max[s](g: class_student_grade(c, s, g))
```

At the end, we will get `{(0, "jerry"), (1, "sherry")}`.
Note that `"jerry"` is the one who got the highest score in class `0` and
`"sherry"` is the one who got the highest score in class `1`.

### Types

Scallop is a statically typed language which employs type inference, which is why
you don't see the type definitions above.
If you want, it is possible to define the type of the relations and even create new type
aliases.
For example,

```
type edge(usize, usize)
```

Defines that the relation `edge` will be a 2-relation and both of the arguments are of
type `usize`, which follows rust's type idiomatic and represents an unsigned 64 bit numbers
(in a 64-bit system).

Scallop supports the following primitive types:

- Signed Integers: `i8`, `i16`, `i32`, `i64`, `i128`, `isize`
- Unsigned Integers: `u8`, `u16`, `u32`, `u64`, `u128`, `usize`
- Floating Points: `f32`, `f64`
- Boolean: `bool`
- Character: `char`
- String:
  - `&str` (static string which could only be used in static Scallop compiler);
  - `String`

Some example type definition includes

```
type edge(usize, usize)
type obj_color(usize, String) // object is represented by a number, usize, and color is represented as string
type empty() // 0-arity relation
type binary_expr(usize, String, usize, usize) // expr id, operator, lhs expr id, rhs expr id
```

### Subtype and Type Alias

The following snippet shows how you can define subtype.

```
type Symbol <: usize
type ObjectId <: usize
```

### Input/Output

It is possible to define a relation is an input relation that can be loaded from files.

```
@file("example/input_csv/edge.csv")
type edge(usize, usize)
```

Note that in this case it is essential to define the type of the relation.
When loading `.csv` files, we accept extra loading options:

- deliminator: `@file("FILE.csv", deliminator = "\t")` with deliminator set to a tab (`'\t'`)
- has header: `@file("FILE.csv", has_header = true)`. It is default to `false`
- has probability: `@file("FILE.csv", has_probability = true)`. When set to `true`, the first
  column of the CSV file will be treated as the probability of each tuple.
