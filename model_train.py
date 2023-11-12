import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

save_directory = 'Generated Data/'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

class LinearRegression(tf.Module):
    def __init__(self, input_size, output_size):
        self.W = tf.Variable(tf.random.normal([input_size, output_size]), name='weights')
        self.b = tf.Variable(tf.zeros([output_size]), name='bias')

    def __call__(self, X):
        return tf.matmul(X, self.W) + self.b

def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
    
df = pd.read_csv('data_daily.csv', parse_dates=['# Date'], index_col=None)  # Load the dataset
df.rename(columns={"# Date": "Date"}, inplace=True)                         # Rename Date Column

plt.figure(figsize=(15, 7))
sns.scatterplot(x=df['Date'], y=df['Receipt_Count'], color='#93E9BE', label='Actual Daily Receipt Count')
plt.title('Comparison of Daily Receipt Trends of Year 2021')
plt.xlabel('Months in the Year 2021')
plt.ylabel('Number of Receipts')
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.legend()
plt.savefig(os.path.join(save_directory, 'data_visualization.png'), bbox_inches='tight')
plt.show()


X_np = np.arange(0, 365) / 364.0
y_np = df['Receipt_Count'].to_numpy() / np.max(df['Receipt_Count'])

train_size = int(0.8 * len(X_np))

X_train, X_test = X_np[:train_size], X_np[train_size:]
y_train, y_test = y_np[:train_size], y_np[train_size:]
date_train, date_test = df['Date'][:train_size], df['Date'][train_size:]

X = tf.constant(X_train, dtype=tf.float32)[:, tf.newaxis]
y = tf.constant(y_train, dtype=tf.float32)[:, tf.newaxis]

# Initialize the model
input_size = 1
output_size = 1
model = LinearRegression(input_size, output_size)

optimizer = tf.optimizers.SGD(learning_rate=0.1)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        y_predicted = model(X)
        loss = mean_squared_error(y, y_predicted)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if (epoch + 1) % 100 == 0:
        print("Epoch: {}, Loss: {:.5f}".format(epoch + 1, loss.numpy()))

# Convert the TensorFlow tensor to a NumPy array
predicted_y_train_np = model(X).numpy()

# Plot the training data prediction and the actual data using Seaborn
plt.figure(figsize=(15, 7))
sns.scatterplot(x=date_train, y=y_train.flatten(), color='#93E9BE', label='Actual Daily Receipt Count')
sns.lineplot(x=date_train, y=predicted_y_train_np.flatten(), color='#FF5349', label='Model Prediction over Training Data', linestyle='dashed')
plt.title('Comparison of Daily Receipt Trends: Model Predictions on Training Data vs Actual Counts in the Year 2021')
plt.xlabel('Months in the Year 2021')
plt.ylabel('Number of Receipts')
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.legend()
plt.savefig(os.path.join(save_directory, 'training_visualization.png'), bbox_inches='tight')
plt.show()

X_test_tensor = tf.constant(X_test, dtype=tf.float32)[:, tf.newaxis]
y_test_tensor = tf.constant(y_test, dtype=tf.float32)[:, tf.newaxis]

predicted_y = model(X_test_tensor)
val_loss = mean_squared_error(y_test_tensor, predicted_y).numpy()

print("Validation Loss:", round(val_loss, 4))

# Save the model
tf.saved_model.save(model, 'Receipts_model')

# Convert the TensorFlow tensor to a NumPy array
predicted_y_np = predicted_y.numpy()

# Plot the test data prediction and the actual data using Seaborn
plt.figure(figsize=(15, 7))
sns.scatterplot(x=date_test, y=y_test.flatten(), color='#93E9BE', label='Daily Receipts Count')
sns.lineplot(x=date_test, y=predicted_y_np.flatten(), color='#FF5349', label='Model Prediction over Testing Data', linestyle='dashed')
plt.title('Comparison of Daily Receipt Trends: Model Predictions on Test Data vs Actual Counts')
plt.xlabel('Date (Daily Intervals)')
plt.ylabel('Number of Receipts')
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.legend()
plt.savefig(os.path.join(save_directory, 'testing_visualization.png'), bbox_inches='tight')
plt.show()