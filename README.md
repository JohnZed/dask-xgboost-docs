# RAPIDS, Dask, and XGBoost

Dask and XGboost are two of the most popular open source packages for data
processing and machine learning, respectively.  RAPIDS augments them to support
accelerated computation across many GPUs on one or several nodes.

RAPIDS integrates cuDF, a GPU-based dataframe implementation, with a CUDA-aware
version of Dask to accelerate data loading and preparation. These can pass data
seamlessly into XGBoost for model training, while reducing the amount of
unncessary data copying.

# 

## Dask [Dask](https://dask.org "Dask: Scalable Analytics in Python") is a
framework for flexibly scaling computational graphs in Python on a single node
or across a cluster.

It implements several distributed datastructures - most importantly a
distributed, Pandas-style DataFrame that elastically scales over many workers.

```python
import dask.dataframe as dd
df = dd.read_csv('2014-*.csv')
df.head()

     x  y
  0  1  a
  1  2  b
  2  3  c
  3  4  a

df2 = df[df.y == 'a'].x + 1
```

## cuDF
[RAPIDS](https://rapids.ai "RAPIDS: Open GPU Data Science") is a collection of open source software libraries aimed at accelerating data science applications end-to-end with GPUs.

Using a Pandas-compatible API, users can read in data, and perform various analytical tasks on data on GPU.

```python
import cudf
gdf = cudf.read_csv('path/to/file.csv')
for column in gdf.columns:
    print(gdf[column].mean())
```

## Dask-cuDF
[Dask-cuDF](https://github.com/rapidsai/dask-cudf "Dask-cuDF: Partitioned GPU-backed DataFrame") implements a distributed CUDA DataFrame much like the Dask DataFrame implements a distributed Pandas DataFrame.

## XGBoost
The [RAPIDS Fork of XGBoost](https://github.com/rapidsai/xgboost "RAPIDS XGBoost") enables XGBoost with cuDF: a user may directly pass a cuDF object into XGBoost for training, prediction, etc.

## Dask-XGBoost

The [RAPIDS Fork of Dask-XGBoost](https://github.com/rapidsai/dask-xgboost/
"RAPIDS Dask-XGBoost") enables XGBoost with the distributed CUDA DataFrame via
Dask-cuDF. A user may pass Dask-XGBoost a reference to a distributed cuDF
object, and start a training session over an entire cluster from Python.

# Building a Dask-cuDF-XGBoost pipeline

Training an XGboost model across multiple GPUs with Dask-XGBoost requires 3 main steps:
   (1) Create a CUDA-aware Dask instance, with one worker per GPU
   (2) Load data into a Dask-CuDF dataframe, distributed across GPUs
   (3) Call XGboost's distributed training functions


## (1) Create a CUDA-aware Dask instance

Setting up the Dask instance to use all GPUs on a single machine is
straightforward:

```python
import cudf
import dask
import dask_cudf
import dask_xgboost

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

import subprocess

## XXX: Can we do socket.gethostbyname(socket.gethostname()) or something like that? Seems a little distracting
# First, find our host's IP address
cmd = "hostname --all-ip-addresses"
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
IPADDR = str(output.decode()).split()[0]

# Now launch a Dask-CUDA cluster on the local host
cluster = LocalCUDACluster(ip=IPADDR)
client = Client(cluster)
client
```

Note the use of `from dask_cuda import LocalCUDACluster`. [Dask-CUDA](https://github.com/rapidsai/dask-cuda) is a lightweight set of utilities useful for setting up a Dask cluster. These calls instantiate a Dask-cuDF cluster in a single node environment. To instantiate a multi-node Dask-cuDF cluster, a user must use `dask-scheduler` and `dask-cuda-worker`. See the [Dask distributed documentation][http://distributed.dask.org/en/latest/setup.html] for more details.

Once a `client` is available, a user may use the client to run tasks across the workers:

```python
def do_computation():
  return 1

client.run(do_computation)
```

Alternatively, a user may employ `dask_cudf` to distribute both data and computation over the workers:

## _TODO: Not necessarily in this chunk, but I think one of the big missing pieces to me is understanding how
## dask_cudf distributes data across the workers / nodes / GPUs. I think this will be a common question for
## customers too_
##
## _Would it be better to show an example that uses dask to do the data loading?_

```python
pdf = pd.DataFrame(
  {"x": np.random.randint(0, 5, size=10000),
  "y": np.random.normal(size=10000)})           # Construct Pandas structure in CPU memory

gdf = cudf.DataFrame.from_pandas(pdf)           # CuDF transfers the data to local GPU memory (??)
ddf = dask_cudf.from_cudf(gdf, npartitions=5)   # Dask-CuDF distributes the data across all GPU workers (??)

ddf = ddf.groupby("x").mean()
```

# Using Dask-XGBoost
There are two functions of interest with Dask-XGBoost:

1. `dask_xgboost.train`
2. `dask_xgboost.predict`

The documentation for `dask_xgboost.train` provides the clearest explanation:

```python
help(dask_xgboost.train)

Help on function train in module dask_xgboost.core:

  train(client, params, data, labels, dmatrix_kwargs={}, **kwargs)
      Train an XGBoost model on a Dask Cluster
      
      This starts XGBoost on all Dask workers, moves input data to those workers,
      and then calls ``xgboost.train`` on the inputs.
      
      Parameters
      ----------
      client: dask.distributed.Client
      params: dict
          Parameters to give to XGBoost (see xgb.Booster.train)
      data: dask array or dask.dataframe
      labels: dask.array or dask.dataframe
      dmatrix_kwargs: Keywords to give to Xgboost DMatrix
      **kwargs: Keywords to give to XGBoost train
      
      Examples
      --------
      >>> client = Client('scheduler-address:8786')  # doctest: +SKIP
      >>> data = dd.read_csv('s3://...')  # doctest: +SKIP
      >>> labels = data['outcome']  # doctest: +SKIP
      >>> del data['outcome']  # doctest: +SKIP
      >>> train(client, params, data, labels, **normal_kwargs)  # doctest: +SKIP
      <xgboost.core.Booster object at ...>
      
      See Also
      --------
      predict
```

An example:

```python
params = {
  'num_rounds':   100,
  'max_depth':    8,
  'max_leaves':   2**8,
  'n_gpus':       1,                  # This will set the number of GPUs, per-worker
  'tree_method':  'gpu_hist',
  'objective':    'reg:squarederror',
  'grow_policy':  'lossguide'
}

bst = dask_xgboost.train(client, params, x_train, y_train, num_boost_round=params['num_rounds'])
``` 

1. `client`: the `dask.distributed.Client`
2. `params`: the training parameters for XGBoost. Note that it is a requirement to set `'n_gpus': 1`, as it tells Dask-cuDF that each worker will have a single GPU to perform coordinated computation
3. `x_train`: an instance of `dask_cudf.DataFrame` containing the data to be trained
4. `y_train`: an instance of `dask_cudf.Series` containing the labels for the training data
5. `num_boost_round=params['num_rounds']`: a specification on the number of boosting rounds for the training session

Likewise, `dask_xgboost.predict` is here:

```python
help(dask_xgboost.predict)

Help on function predict in module dask_xgboost.core:
  
  predict(client, model, data)
      Distributed prediction with XGBoost
      
      Parameters
      ----------
      client: dask.distributed.Client
      model: xgboost.Booster
      data: dask array or dataframe
      
      Examples
      --------
      >>> client = Client('scheduler-address:8786')  # doctest: +SKIP
      >>> test_data = dd.read_csv('s3://...')  # doctest: +SKIP
      >>> model
      <xgboost.core.Booster object at ...>
      
      >>> predictions = predict(client, model, test_data)  # doctest: +SKIP
      
      Returns
      -------
      Dask.dataframe or dask.array, depending on the input data type
      
      See Also
      --------
      train
```

An example:

```python
pred = dask_xgboost.predict(client, bst, x_test)
test = dask.dataframe.multi.concat([pred], axis=1)

test['squared_error'] = (test[0] - y_test['x'])**2
rmse = np.sqrt(test.squared_error.mean().compute())
```

1. `client`: the `dask.distributed.Client`
2. `bst`: the Booster produced by the XGBoost training session
3. `x_test`: an instance of `dask_cudf.DataFrame` containing the data to be inferenced (acquire predictions)

`pred` will be an instance of `dask_cudf.Series`. We can use `dask.dataframe.multi.concat` to construct a `dask_cudf.DataFrame` by concatenating the list of `dask_cudf.Series` instances (`[pred]`). `test` is a `dask_cudf.DataFrame` object with a single column named `0` (e.g.) `test[0]` returns `pred`.

# This is a bit confusing... maybe just needs a link to dask docs explaining the use of compute? Wouldn't be bad to mention
# lazy-computation here
Additionally, the root-mean-squared-error (RMSE) can be computed by constructing a new column and assigning to it the value of the difference between predicted and labeled values squared. This is encoded in the assignment `test['squared_error'] = (test[0] - y_test['x'])**2`. Finally, the mean can be computed by using an aggregator from the `dask_cudf` API. The entire computation is initiated via `.compute()`; finally, we take the square-root of the result, leaving us with `rmse = np.sqrt(test.squared_error.mean().compute())`. Note: `.squared_error` is an accessor for `test[squared_error]`.
