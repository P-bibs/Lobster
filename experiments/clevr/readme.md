# CLEVR

![CLEVR_train_000013.png](docs/CLEVR_train_000013.png)

In this experiment we test Scallop's reasoning ability for the [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) dataset.
The CLEVR dataset contains images that look like above: a table with simple objects on the top of it.
At the same time, the dataset asks questions about the image such as "how many cylinders are there?" (Answer: 5) and "what is the shape of the grey object to the right of the green cylinder?" (Answer: cylinder).

Internally, both the image and the natural language (NL) question are represented in a structured form.
The image is represented as a *Scene Graph*, and the NL question is represented as *Programmatic Query*.
We will introduce how these two structures are represented with Scallop and how they are integrated together as a whole.

## Scene Graph Representation

With Scallop, the scene graph can be represented as a probabilistic database.
For example, there are 10 objects in the above image, marked by integer 0 to 9.

``` scl
rel obj = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
```

Additionally, each object will be associated with a color (for demonstration purpose, assuming object IDs are sorted from left to right).
Other properties like material (`"metal"`, `"rubber"`) and shape (`"cube"`, `"cylinder"`) will be represented in a similar manner.

``` scl
rel color = {
  (0, "red"),
  (1, "brown"),
  (2, "blue"),
  (3, "green"),
  (4, "yellow"),
  // ...
}
```

## Programmatic Query

It turns out that all programmatic query can be represented as structured facts in Scallop and in this case the Scallop program can serve as a differentiable probabilistic interpreter for the query.

Let's suppose we want to ask a question

> How many cylinders are there in the scene?

This question can be represented as a functional program in the following way:

``` scl
count(shape(scene(), "cylinder"))
```

where the `scene()` function returns the set of all objects in the scene,
the `shape(..., "cylinder")` acts like a filter function that keeps all cylinders in the input set,
and the `count(...)` counts the number of objects in the set.

With Scallop, this program will be represented by a set of structured facts:

``` scl
import "scl/clevr_eval.scl" // We import the CLEVR interpreter

rel scene_expr = {0}                  // expr 0 is a scene expression
rel shape_expr = {(1, 0, "cylinder")} // expr 1 is a shape expression that takes in expr 0 and "cylinder" as input
rel count_expr = {(2, 1)}             // expr 2 is a count expression that takes expr 1 as input
rel root_expr = {2}                   // expr 2 is the root expression
```

If executed with the scene graph information, this program will return `5` as the final result as there are 5 cylinders in the scene.
The full Scallop implementation of the CLEVR DSL interpreter is written in [`scl/clevr_eval.scl`](scl/clevr_eval.scl).
If you wish to see one concrete example of how a real question and scene graph pair is represented in the dataset, checkout [`scl/train_sample_0.scl`](scl/train_sample_0.scl) and run it with

```
$ scli scl/train_sample_0.scl
```

## Running Expriments

To run the experiments, make sure that you have downloaded the `CLEVR_v1.0` dataset from the [official website](https://cs.stanford.edu/people/jcjohns/clevr/).
The dataset should be extracted to (starting from the root of Scallop directory) `[SCALLOP_V2_DIR]/experiments/data/CLEVR` and you should see the following folder structure.

```
[SCALLOP_V2_DIR]/experiments/data/CLEVR
> images/
  > test/
    > CLEVR_test_000000.png
    > ...
  > train/...
  > val/...
> questions/...
> scenes/...
> COPYRIGHT.txt
...
```

> The following is still work in progress, please don't follow the instructions yet.

Then, you can get started by running

```
$ python run_vision_only.py
```

This is the experiment where we fix the programmatic query (the structured facts representing the query will be passed as deterministic facts).
And only train the vision components recognizing the
