"""
Dự án cá nhân đầu tay dùng git, neovim, ML
Vừa học ML, git, neovim vừa làm
Mong thành công

Chủ đề: Dự đoán đột quỵ (ChatGPT gợi ý)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in the dataset
df = pd.read_csv("archive/diabetes.csv")
print(df.head())
