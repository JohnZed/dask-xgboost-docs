# RAPIDS, Dask, and XGBoost

## Dask
[Dask](https://dask.org "Dask: Scalable Analytics in Python") is a framework for flexibly scaling computation in Python.

It implements a distributed Pandas DataFrame to elastically scale over many workers.

```python
import dask.dataframe as dd
df = dd.read_csv('2014-*.csv')
df.head()

     x  y
  0  1  a
  1  2  b
  2  3  c
  3  4  a
  4  5  b
  5  6  c

df2 = df[df.y == 'a'].x + 1
```

## cuDF
[RAPIDS](https://rapids.ai "RAPIDS: Open GPU Data Science") is a collection of open source software libraries aimed at accelerating data science applications end-to-end.

Much like Pandas, users can read in data, and perform various analytical tasks in parity with Pandas.

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
The [RAPIDS Fork of Dask-XGBoost](https://github.com/rapidsai/dask-xgboost/ "RAPIDS Dask-XGBoost") enables XGBoost with the distributed CUDA DataFrame via Dask-cuDF. A user may pass Dask-XGBoost a reference to a distributed cuDF object, and start a training session over an entire cluster from Python.

# Using Dask-cuDF
A user may instantiate a Dask-cuDF cluster like this:

```python
import cudf
import dask_cudf
import dask_xgboost
from dask_cuda import LocalCUDACluster

import subprocess

cmd = "hostname --all-ip-addresses"
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
IPADDR = str(output.decode()).split()[0]

cluster = LocalCUDACluster(ip=IPADDR)
client = Client(cluster)
client
```

Note the use of `from dask_cuda import LocalCUDACluster`. [Dask-CUDA](https://github.com/rapidsai/dask-cuda) is a lightweight set of utilities useful for setting up a Dask cluster.

Once a `client` is available, a user may commission the client to perform tasks:

```python
def foo():
  return

client.run(foo)
```

Alternatively, a user may employ `dask_cudf`:

```python
pdf = pd.DataFrame(
  {"x": np.random.randint(0, 5, size=10000), "y": np.random.normal(size=10000)})

gdf = cudf.DataFrame.from_pandas(pdf)

ddf = dask_cudf.from_cudf(gdf, npartitions=5)

ddf = ddf.groupby("x").mean()
```

# Using Dask-XGBoost
There are two functions of interest with Dask-XGBoost:

1. `dask_xgboost.train`
2. `dask_xgboost.predict`

The documentation for `dask_xgboost.train` is this:

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
  'n_gpus':       1,
  'tree_method':  'gpu_hist',
  'objective':    'binary:logistic',
  'grow_policy':  'lossguide'
}

bst = dask_xgboost.train(client, params, ddf, labels, num_boost_round=params['num_rounds'])
``` 

1. `client`: the `dask.distributed.Client`
2. `params`: the training parameters for XGBoost. Note that it is a requirement to set `'n_gpus': 1`, as it tells Dask-cuDF that each worker will have a single GPU to perform coordinated computation
3. `ddf`: an instance of `dask_cudf.DataFrame` containing the data to be trained
4. `labels`: an instance of `dask_cudf.Series` containing the labels for the training data
5. `num_boost_round=params['num_rounds']`: a specification on the number of boosting rounds for the training session