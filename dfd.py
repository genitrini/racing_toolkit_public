import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Example data
x = df['steer'][(df['lap'] ==1) & (df['perc_space'] >=7) & (df['perc_space'] <=12)].to_numpy()
y = df['acc_vert'][(df['lap'] ==1) & (df['perc_space'] >=7) & (df['perc_space'] <=12)].to_numpy()

# Add intercept
X = sm.add_constant(x)

# Fit linear regression
model = sm.OLS(y, X).fit()

# Regression results
print(model.summary())

# Coefficients
print("Intercept:", model.params[0])
print("Slope:", model.params[1])
print("R-squared:", model.rsquared)
print("p-value for slope:", model.pvalues[1])
print("95% CI:", model.conf_int())

# Plot
plt.scatter(x, y, label="Observed data")
plt.plot(x, model.predict(X), color="red", label="Regression line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()