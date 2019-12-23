from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from time import sleep
from datetime import datetime
from source import read_data,plot,preprocessing,train_test,training_model,classifier,compute
from source import main as a

def read_csv_file():
	sleep(1)
	return read_data

def mat_plot():
	sleep(1)
	return plot

def preprocess():
        sleep(1)
        return preprocessing


def training_dataset():
        sleep(1)
        return train_test


def train_model():
        sleep(1)
        return training_model


def accuracy_of_data():
        sleep(1)
        return a.accuracy


def classify():
        sleep(1)
        return classifier


def computing():
        sleep(1)
        return compute


with DAG('MLOps_dag',description='First DAG', schedule_interval='*/10 * * * *',start_date=datetime(2019,12,23),catchup=False) as dag:
	start_task=DummyOperator(task_id='Start',retries=3)
	read_csv=PythonOperator(task_id='Read_CSV',python_callable=read_csv_file)
	plot=PythonOperator(task_id='Plotting_graph',python_callable=mat_plot)
	preprocessing=PythonOperator(task_id='Preprocess_data',python_callable=preprocess)
	train_test=PythonOperator(task_id='Train_dataset',python_callable=training_dataset)
	training_model=PythonOperator(task_id='Train_model',python_callable=train_model)
	accuracy=PythonOperator(task_id='Check_Accuracy',python_callable=accuracy_of_data)
	classifier=PythonOperator(task_id='Classify',python_callable=classify)
	compute=PythonOperator(task_id='Compute',python_callable=computing)

	start_task >> read_csv >> plot
	start_task >> read_csv >> preprocessing >> train_test >> training_model 
	start_task >> read_csv >> classifier >> compute
