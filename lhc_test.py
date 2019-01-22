import pyDOE
import matplotlib.pyplot as plt


lh = pyDOE.lhs(2, samples=20, criterion='centermaximin')
#“center” or “c”: center the points within the sampling intervals
#“maximin” or “m”: maximize the minimum distance between points, but place the point in a randomized location within its interval
#“centermaximin” or “cm”: same as “maximin”, but centered within the intervals
#“correlation” or “corr”: minimize the maximum correlation coefficient



#%
x = []
y = []
for entry in lh:
    x.append(entry[0])
    y.append(entry[1])


plt.scatter(x,y)
plt.show()
