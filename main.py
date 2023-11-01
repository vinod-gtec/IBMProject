import pandas as pd
from sklearn.model_selection import train_test_split
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import  matplotlib.pyplot as plt
from datetime import datetime


pd.set_option("display.max_columns", None)
df = pd.read_csv("Sales.csv")
q1 = df["Newspaper"].quantile(0.25)
q3 = df["Newspaper"].quantile(0.75)
thr = 1.5
iqr = q3-q1
lower = q1-thr*iqr
upp = q3+thr*iqr
mean_value = df["Newspaper"].mean()
df["Newspaper"][(df["Newspaper"] < lower) | (df["Newspaper"] > upp)] = mean_value

date_values = pd.date_range(start='2023-10-28', periods=len(df), freq='D')
df["dates"] = date_values
df = df.rename(columns={"dates": "ds"})
x = df.drop("Sales", axis=1)
y = df["Sales"]
print(df.head())
plt.boxplot(df["TV"])
plt.show()
plt.boxplot(df["Radio"])
plt.show()
plt.boxplot(df["Newspaper"])
plt.show()
x_values = list(range(1, len(df["TV"])+1))
plt.scatter(x_values, df["TV"], label="datapoints", color="blue")

# there are no categorical values so need of onehot encoder
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=3)
train_data = pd.DataFrame({"ds": X_train["ds"], "y": y_train, 'TV': X_train['TV'], 'Radio': X_train['Radio'], 'Newspaper': X_train['Newspaper']})

# Step 1: Load your dataset with dates, sales, and additional features

# Step 2: Data preprocessing
data = df.rename(columns={'Date': 'ds', 'Sales': 'y', 'TV': 'TV', 'Radio': 'Radio', 'Newspaper': 'Newspaper'})

# Step 3: Initialize Prophet model with additional regressors
model = Prophet()
model.add_regressor('TV')
model.add_regressor('Radio')
model.add_regressor('Newspaper')

# Step 4: Fit the model with the entire dataset
model.fit(train_data)

# Step 5: Create a dataframe for future dates and features for prediction
future = model.make_future_dataframe(periods=len(X_test))

# Add feature data for the test set
future['TV'] = float(input("Enter the amount to be spent on Tv"))
future['Radio'] = float(input("Enter the amount to be spent on Radio"))
future['Newspaper'] = float(input("Enter the amount to be spent on Newspaper"))
# Step 6: Make predictions for the combined dataframe
forecast = model.predict(future)


# Step 7: Visualize the predictions
fig = model.plot(forecast)
plt.show()

# Prompt the user for a date input
date_string = input("Enter a date (YYYY-MM-DD): ")

try:
    # Attempt to convert the user input to a date
    user_date = datetime.strptime(date_string, "%Y-%m-%d")
    print("You entered:", user_date)
except ValueError:
    print("Invalid date format. Please use YYYY-MM-DD format.")
specific_date = user_date

prediction = forecast.loc[forecast["ds"] == specific_date]['yhat'].values[0]
print("  ")
print("THE PREDICTED SALES ON THE DATE", prediction+0.5)
print(" ")
components = model.plot_components(forecast)

#result
result = cross_validation(model, horizon="1 days", period="30 days", initial="90 days")
metrics = performance_metrics(result)
print(" ")
print(" ")
print("THE PREDICTED SALES ON THE DATE", prediction+0.5)
print("Here is the evaluation output of various metrics")
print(metrics)
