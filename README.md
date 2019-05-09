# RAPIDS, Dask, and XGBoost

## Dask
[Dask](https://dask.org "Dask: Scalable Analytics in Python") is a framework for flexibly scaling computation in Python.

It implements a distributed Pandas DataFrame to elastically scale over many workers.

```python
>>> import dask.dataframe as dd
>>> df = dd.read_csv('2014-*.csv')
>>> df.head()
   x  y
0  1  a
1  2  b
2  3  c
3  4  a
4  5  b
5  6  c

>>> df2 = df[df.y == 'a'].x + 1
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
The [RAPIDS Fork of Dask-XGBoost](https://github.com/rapidsai/dask-xgboost/ "RAPIDS Dask-XGBoost") enables XGBoost with the distributed CUDA DataFrame via Dask-cuDF.