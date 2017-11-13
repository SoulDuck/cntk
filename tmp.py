import numpy as np
import pandas as pd
print np.linspace(0, 100, 10000, dtype=np.float32)
print len(np.linspace(0, 100, 10000, dtype=np.float32))


s = pd.Series(np.nan, index=[49,48,47,46,45, 1, 2, 3, 4, 5])
#s = pd.Series(np.nan, index=['a','b','c','d','e', 1, 2, 3, 4, 5])
print s.iloc[:3]
print s.loc[:3]
print s.ix[:3]