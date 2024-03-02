# Convert from celcius to F
import pandas as pd
import seaborn as sb

# I need this to show the seaborn graphs
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

data = pd.read_csv("dataset.csv")

# Seabron seems to use matplotlib as a dependency or something
sb.scatterplot(
        x="celsius",
        y="fahrenheit",
        data=data,
        hue="fahrenheit",
        palette="coolwarm")

# Characteristic (x), label (y)
x = data["celsius"]
y = data["fahrenheit"]
# the required data is [[x], [x1], [x2], ...[xn]]
x_shaped = x.values.reshape(-1, 1)
y_shaped = y.values.reshape(-1, 1)

# Create the model
model = LinearRegression()

# Train the model
# I don't have to create the freaking line... the whole model will do it for me
# THIS IS FUCKING AMAZING
model.fit(x_shaped, y_shaped)

# Do a prediction
p = model.predict([[123]])

print(p)

# Show the plot
# plt.show()
