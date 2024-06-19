For now this is just a quick note to get people up to speed on the latest changes. TBD: more real docs.

As well as `Parameter` nodes, there are now `Observation` nodes.
Parameter nodes are for unknown values that we want the optimizer to find. They are not named, and their creation is generally encapsulated because the parameterization is an implementation detail of the model.
Observation nodes are for values that are known at any point in time, but will change rapidly and interactively (dragging a point, moving a slider). They are named (eg, `ops.observation("mouse_x")`).

After you create a node graph with parameters and (optionally) observations, you need to:

* create a `loss.Loss` object, giving it a node representing the loss to minimize
* use the loss object's `register_output` method to register any nodes that you will later want to evaluate. This is so that those nodes can be compiled along with the loss node as a coherent unit. Note: to ensure this coherency, the parameters and observations referenced by these nodes must be a subset of those referenced by the loss node. 
* create a `optimizer.Optimizer` node, giving it the loss object. This is a somewhat awkward separation because `Loss` doesn't know anything about JAX whereas everything in `Optimizer` is JAX-specific.
* you can now call `optimize()` on the optimizer any number of times without triggering any recompilation, passing in a dictionary mapping the observation names to float values: eg `opt.optimize({"mouse_x": 123.0, "mouse_y": 456.0})`. It will return a `Solution` object.
* finally, you can use `eval` on the `Solution` object to get a concrete value back for any node that you originally registered as an output.

The node objects are immutable, as are the Solution objects. The solution objects store both the observation values they were created with, and the discovered optimal parameter values, and will use both of these in the evaluation. There's no problem creating multiple different solutions from the same Optimizer and evaluating the same node objects with them; the results will always be consistent.

See test.py for a very simple demonstration of this.