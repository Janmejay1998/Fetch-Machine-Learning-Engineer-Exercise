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
