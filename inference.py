import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import os

save_directory = 'Generated Data/'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

#Define the model
class LinearRegression(tf.Module):
    def __init__(self, input_size, output_size):
        self.W = tf.Variable(tf.random.normal([input_size, output_size]), name='weights')
        self.b = tf.Variable(tf.zeros([output_size]), name='bias')

    def __call__(self, X):
        return tf.matmul(X, self.W) + self.b
    
data = pd.read_csv('data_daily.csv', parse_dates=['# Date'], index_col=None)  # Load the dataset
data.rename(columns={"# Date": "Date"}, inplace=True)                         # Rename Date Column

date_range = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
inference_data = pd.DataFrame({'Date': date_range})

initial_date = np.datetime64('2021-01-01')
num_of_days = (inference_data['Date'] - initial_date).dt.days / 364

X = tf.convert_to_tensor(num_of_days, dtype=tf.float32)[:, tf.newaxis]

# Initialize the model
input_size = 1
output_size = 1
model = LinearRegression(input_size, output_size)
loaded_model = tf.saved_model.load('Receipts_model')
model.W.assign(loaded_model.W)
model.b.assign(loaded_model.b)

model(tf.constant([[0.0]], dtype=tf.float32))  # Run the model to initialize variables

y_predict = model(X)  # Run the model to make predictions
y_predict *= np.max(data['Receipt_Count'].to_numpy())  

# Assuming y_predict_tensor is your TensorFlow tensor
df = pd.DataFrame({
    'Date': pd.date_range(start='2022-01-01', periods=365),
    'Receipts': y_predict.numpy().squeeze()
}).set_index('Date')

# Resample by month and sum up the receipts, then save to CSV
df.resample('M').sum().to_csv(os.path.join(save_directory, 'predicted_receipts.csv'), index=True)

# Convert TensorFlow tensor to NumPy array
y_predict_np = y_predict.numpy()

# Plot the results using Seaborn
plt.figure(figsize=(15, 7))
sns.scatterplot(x=data['Date'], y=data['Receipt_Count'], color='#93E9BE', label='Actual Daily Receipt Count')
sns.lineplot(x=inference_data['Date'], y=y_predict_np.squeeze(), color='#FF5349', label='Model Prediction over Year 2022')
plt.title('Comparison of Model Predictions and Actual Daily Receipt Counts (2021-2022)')
plt.xlabel('Date (Year 2021 - Year 2022)')
plt.ylabel('Number of Receipts')
plt.grid(True, which='both', linestyle='-', linewidth=0.5) 
plt.legend()
plt.savefig(os.path.join(save_directory, 'inference_visualization.png'), bbox_inches='tight')
plt.show()