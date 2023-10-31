

## 数据集缩放

```python
import numpy as np
w_rgb, h_rgb = (1920,1080)
w_gray, h_gray = (640, 512)
# 归一化数据集
rgb_points=rgb_data_04
gray_points=gray_data_04
rgb_points = rgb_points/np.array([w_rgb, h_rgb])
rgb_points = rgb_points - np.array([0.5, 0.5])
gray_points = gray_points/np.array([w_gray, h_gray])
gray_scale = gray_points - np.array([0.5, 0.5])
gray_points_scale
```

## 数据预测

```python
import joblib
from sklearn.preprocessing import PolynomialFeatures
# 从文件中加载模型
loaded_model = joblib.load('Quadratic_terms_regression_model.pkl')
# 使用加载的模型进行预测
#缩放后的数据一定要先使用 PolynomialFeatures 进行特征转换，将一维特征数据转换为包含一次和二次项的多项式特征
poly_data= poly.fit_transform([[5, 10],[20,78]])
predictions = loaded_model.predict(poly_data)  
predictions
```



## 数据还原

```python
#数据还原
rgb_poly_raw = predictions + np.array([0.5, 0.5])
# rgb_pred_linear
rgb_poly_raw = rgb_poly_raw*np.array([w_rgb, h_rgb])
rgb_poly_raw
```

