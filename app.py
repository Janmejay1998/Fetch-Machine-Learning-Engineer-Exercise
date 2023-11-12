from flask import Flask, render_template, request
from wtforms import Form, DateField
from wtforms.validators import DataRequired
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import plotly.graph_objs as go
from datetime import datetime

app = Flask(__name__)

save_directory = 'Generated Data/'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Define the Linear Regression Model
class LinearRegression(tf.Module):
    def __init__(self, input_size, output_size):
        self.W = tf.Variable(tf.random.normal([input_size, output_size]), name='weights')
        self.b = tf.Variable(tf.zeros([output_size]), name='bias')

    def __call__(self, X):
        return tf.matmul(X, self.W) + self.b

# Load the pre-trained model
input_size = 1
output_size = 1
model = LinearRegression(input_size, output_size)
loaded_model = tf.saved_model.load('Receipts_model')
model.W.assign(loaded_model.W)
model.b.assign(loaded_model.b)

# Define the form
class DateForm(Form):
    default_start_date = datetime.strptime('2022-01-01', '%Y-%m-%d').date()
    start_date = DateField('Start Date', format='%Y-%m-%d', default=default_start_date, validators=[DataRequired()])
    end_date = DateField('End Date', format='%Y-%m-%d', validators=[DataRequired()])

# Define the inference route
@app.route('/', methods=['GET', 'POST'])
def index():
    form = DateForm(request.form)

    if request.method == 'POST' and form.validate():
        start_date = form.start_date.data
        end_date = form.end_date.data

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        inference_data = pd.DataFrame({'Date': date_range})

        data = pd.read_csv('data_daily.csv', parse_dates=['# Date'], index_col=None)  # Load the dataset
        data.rename(columns={"# Date": "Date"}, inplace=True)                         # Rename Date Column

        initial_date = np.datetime64('2021-01-01')
        num_of_days = (inference_data['Date'] - initial_date).dt.days / 364

        X = tf.convert_to_tensor(num_of_days, dtype=tf.float32)[:, tf.newaxis]

        y_predict = model(X)
        y_predict *= np.max(data['Receipt_Count'].to_numpy())

        # Convert TensorFlow tensor to NumPy array
        y_predict_np = y_predict.numpy()

        # Assuming y_predict_tensor is your TensorFlow tensor
        df = pd.DataFrame({
            'Date': pd.date_range(start=start_date, end=end_date),
            'Receipts': y_predict_np.squeeze()
        }).set_index('Date')
        
        # Resample by month and sum up the receipts
        df_monthly = df.resample('M').sum()
    
        # Create an interactive line chart using Plotly Express
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=inference_data['Date'], y=y_predict_np.squeeze(), mode='lines', marker=dict(color='#FF5349'), name='Monthly Predicted Receipt Count'))
        fig.update_layout(title_text='Predicted Monthly Receipts Count Over Time')
        fig.update_xaxes(title_text='Time Period')
        fig.update_yaxes(title_text='Number of Receipts')
        # Adding scatter plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Receipt_Count'], mode='markers', marker=dict(color='#93E9BE'), name='Actual Daily Receipt Count'))

        plot_div = fig.to_html(full_html=False)
        return render_template('index.html', form=form, data=df_monthly.to_html(), plot_div=plot_div)

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
