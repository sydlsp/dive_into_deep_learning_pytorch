import numpy as np
import pandas as pd
data = pd.DataFrame({"学号":[1001,1002,1003,1004],
                    "性别":["男","女","女","男"],
                    "学历":["本科","硕士","专科","本科"]})
print(data)
data=pd.get_dummies(data,dummy_na=True)
print(data)