# Fetch-Machine-Learning-Engineer-Exercise

## Directory Structure
```
C:.
|   app.py
|   data_daily.csv
|   Dockerfile
|   inference.py
|   model_train.py
|   README.md
|   requirements.txt
|   
+---Generated Data
|       data_visualization.png
|       inference_visualization.png
|       predicted_receipts.csv
|       testing_visualization.png
|       training_visualization.png
|       
+---Receipts_model
|   |   fingerprint.pb
|   |   saved_model.pb
|   |   
|   +---assets
|   \---variables
|           variables.data-00000-of-00001
|           variables.index
|           
\---templates
        index.html
```        
## Project Structure
- **app.py:** Flask code for starting web service application.
- **data_daily.csv:** Contains data of number of the observed scanned receipts each day for the year 2021.
- **Dockerfile:** Configuration for Docker containerization.
- **inference.py:** Contains code for making inference on 2022 year.
- **model_train.py:** Contains code for training model over 2021 year data.
- **README.md:** Project documentation.
- **requirements.txt:** List of Python dependencies.
- **Generated Data:**
  - **data_visualization.png:** Showing data visualization of data_daily.csv.
  - **inference_visualization.png:** Showing inference visualization.
  - **predicted_receipts.csv:** Storing predictions values of each month of year 2022.
  - **testing_visualization.png:** Showing test visualization.
  - **training_visualization.png:** Showing train visualization.
- **Reciepts_model:** Saved trained model for using in inference.
  - **fingerprint.pb**
  - **saved_model.pb**
  - **assets**
  - **variables**
    - **variables.data-00000-of-00001**
    - **variables.index**
- **templates:**
  - **index.html:** Contains flask web service visual design
## Data Visualization
![Display Image](https://github.com/Janmejay1998/Fetch-Machine-Learning-Engineer-Exercise/blob/main/Generated%20Data/data_visualization.png)
![Display Image](https://github.com/Janmejay1998/Fetch-Machine-Learning-Engineer-Exercise/blob/main/Generated%20Data/training_visualization.png)
![Display Image](https://github.com/Janmejay1998/Fetch-Machine-Learning-Engineer-Exercise/blob/main/Generated%20Data/testing_visualization.png)
![Display Image](https://github.com/Janmejay1998/Fetch-Machine-Learning-Engineer-Exercise/blob/main/Generated%20Data/inference_visualization.png)

## Model Architecture

Linear regression is a linear model that assumes a linear relationship between the input variables and the single output variable. The architecture of the model consists of a single layer with two parameters, weights and bias. The weights are initialized randomly and updated during training to minimize the difference between the predicted output and the actual output. The bias is initialized to zero and also updated during training. The model takes an input tensor X and returns the predicted output by multiplying X with the weights and adding the bias. The predicted output is then compared with the actual output to calculate the loss, which is minimized using an optimization algorithm such as gradient descent. The model can be used for both simple and multiple linear regression problems.

## Prediction Table
\begin{table}[]
\begin{tabular}{|l|l|lll}
\cline{1-2}
Date       & Receipts  &  &  &  \\ \cline{1-2}
31-01-2022 & 316706080 &  &  &  \\ \cline{1-2}
28-02-2022 & 291877220 &  &  &  \\ \cline{1-2}
31-03-2022 & 329593470 &  &  &  \\ \cline{1-2}
30-04-2022 & 325408670 &  &  &  \\ \cline{1-2}
31-05-2022 & 342917760 &  &  &  \\ \cline{1-2}
30-06-2022 & 338303140 &  &  &  \\ \cline{1-2}
31-07-2022 & 356242050 &  &  &  \\ \cline{1-2}
31-08-2022 & 363013380 &  &  &  \\ \cline{1-2}
30-09-2022 & 357750500 &  &  &  \\ \cline{1-2}
31-10-2022 & 376337660 &  &  &  \\ \cline{1-2}
30-11-2022 & 370644960 &  &  &  \\ \cline{1-2}
31-12-2022 & 389661950 &  &  &  \\ \cline{1-2}
\end{tabular}
\end{table}
