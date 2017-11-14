## Notes

#### Matplotlib
1. Line plot is better when x axis is time
2. Scatter is better when there is correlation between two variables
3. Histogram is better when we need to see distribution of numerical data
4. Customizations -
    * color
    * labels
    * thickness of line
    * title
    * opacity
    * grid
    * figsize
    * ticks of axis
    * linestyle

#### Dictionary
1. It has 'key' and 'value'
2. Faster than lists
3. Keys have to be immutable objects like string, boolean, float, integer or tuples

#### Iterators
1. Iterable is an object that can return an iterator
2. Iterable: an object with an associated iter() method.
    * ex- list, string and dictionaries
3. Iterator: produces next value with next() method
```python
name = "ronaldo"
it = iter(name)
print(next(it))     # print next iteration
print(*it)          # print remaining iteration
```

### Cleaning data
1. Unclean data:
    * Column name inconsistency like upper-lower case letter or space between words
    * missing data
    * different language
2. Explotary Data Analysis
    * value_counts
    * describe
3. Tidy, Pivot or Concat
4. Missing data:
    * leave as is
    * drop them with dropna()
    * fill missing value with fillna()
    * fill missing values with test statistics like mean

### Manipulating Dataframes
1. Indexing Data Frames
    * Indexing using square brackets
    * Using column attribute and row label
    * Using loc accessor
    * Selecting only some columns