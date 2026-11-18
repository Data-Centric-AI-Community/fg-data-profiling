import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

data = np.random.uniform(size=6)
data[0] = 1e16
df = pd.DataFrame(dict(a=data))
ProfileReport(df, tsmode=False, lazy=False)