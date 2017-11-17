import numpy as np
import pandas as pd
print np.linspace(0, 100, 10000, dtype=np.float32)
print len(np.linspace(0, 100, 10000, dtype=np.float32))
import input_sine

s = pd.Series(np.nan, index=[49,48,47,46,45, 1, 2, 3, 4, 5])
#s = pd.Series(np.nan, index=['a','b','c','d','e', 1, 2, 3, 4, 5])
print s.iloc[:3]
print s.loc[:3]
print s.ix[:3]

N , M = 5,5
X,Y=input_sine.generate_data(np.sin  , np.linspace(0,100,10000 , dtype=np.float32) ,  N, M )

print [0.01]*25 + [0.001]*25 + [0.0001]*25 + [0.00001]*25 + [0.000001]
N = 5
X = np.arange(3*N).reshape(N,3).astype(np.float32) # 6 rows of 3 values
print X