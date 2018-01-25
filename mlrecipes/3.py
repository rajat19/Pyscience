# pylint: disable=E1101
import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_heigth = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist(
    [grey_heigth, lab_height],
    stacked=True,
    color=['r', 'b']
)
plt.show()