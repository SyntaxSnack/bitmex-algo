import pandas as pd
from pandas.io.pytables import array_equivalent

def return_closest(n, round_set):
  return min(round_set, key=lambda x:abs(x-n))

def duplicate_columns(frame, round=False, round_set=[]):
    print("frame", frame)
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []
    for t, v in groups.items():
        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)
        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if(round and (isinstance(ia) and isinstance(ja))):
                    ia = return_closest(ia, round_set)
                    ja = return_closest(ja, round_set)
                if array_equivalent(ia, ja):
                    dups.append(cs[i])
                    break
    return dups