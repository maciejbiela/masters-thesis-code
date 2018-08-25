### Maciej Biela, Master's Thesis code repository
####Image analysis for the identification of foundry details

###### Prerequisites
- OpenCV 3.4.2 with Python Bindings
- NumPy, Matplotlib, imutils (see [`Pipfile`](Pipfile))

###### Run for bucketized angles
To run the comparison against the input data, please run [`split_histogram.py`](split_histogram.py).

In order to run it for bucketized angles, please start:
```python
perform_comparisons([
    'deg_0/',
    'deg_45/',
    'deg_90/',
    'deg_135/',
    'deg_180/',
    'deg_225/',
    'deg_270/',
    'deg_315/'
])
```

Output of this function produces tables for corresponding angles in LaTeX format and various statistics for the results.

###### Run for all input data
To run the comparison against the whole set of input data, please run [`split_histogram.py`](split_histogram.py).

Please start:
```python
perform_comparisons([
    'all/'
])
```

This functions produces the same output variables.

**Important:** Cross-comparison of the whole input set takes significant amount of time (~20 minutes)

##### Master's thesis advisor: prof. dr hab. inż. Marek Skomorowski
##### Master's thesis secondary advisor: dr inż. Michał Markiewicz
##### Jagiellonian University, Cracow 2018