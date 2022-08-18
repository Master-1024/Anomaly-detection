from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import os

from datetime import datetime
from sklearn.ensemble import IsolationForest
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class fetch_data_item(BaseModel):
    customer: Optional[str] = None
    resource_name: str
    metric_name: str
    metric_value: Optional[str] = None
    filterr: Optional[dict] = None

def set_connection():
    token = "xxxx"
    org = "xxxx"
    client = InfluxDBClient(url=str(os.getenv('xxxx')), token=token, org=org)
    return client

def get_dataframe(client,component_name,metric_name,filterr):
    query_filter=''
    query_api = client.query_api()
    if filterr!=None:
        for key, value in filterr.items():
            query_filter += '''\n|> filter(fn: (r) => r["'''+key+'''"] == "'''+value+'''")'''

    q='''from(bucket: "xxxx")
             |> range(start: -20d, stop: now())
             |> filter(fn: (r) => r["_measurement"] == "'''+component_name+'''")
             '''+query_filter+'''
             |> filter(fn: (r) => r["_field"] == "'''+metric_name+'''")
             |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
        '''
    result = query_api.query(query=q)
    results = []
    for table in result:
        for record in table.records:
            results.append({'time':str(record.get_time()),'value':record.get_value()})
    df = pd.DataFrame.from_dict(results)
    return df

@app.post("/train")
def train(item: fetch_data_item):
    resource_name = item.resource_name
    metric_name = item.metric_name
    filterr = item.filterr
    filterr_file_name = ''
    client = set_connection()
    path = str(os.getenv('PARQUET_FILEPATH'))+'/model_univariate/'
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
    df = get_dataframe(client,resource_name,metric_name,filterr)
    if not df.empty:
        model=IsolationForest(n_estimators=100,max_samples='auto',contamination=0.1)
        model.fit(df[['value']])
        df['scores'] = model.decision_function(df[['value']])
        df['anomaly_score'] = model.predict(df[['value']])
        upper = df[df['anomaly_score']==-1]['value'].mean() + 1.5*df[df['anomaly_score']==-1]['value'].std()
        lower = df[df['anomaly_score']==-1]['value'].mean() - 1.5*df[df['anomaly_score']==-1]['value'].std()
        if filterr==None:
            model_filename = path+str(resource_name)+'_'+str(metric_name)+'_MODEL.sav'
            upper_lower_filename = path+str(resource_name)+'_'+str(metric_name)+'_THRESHOLD.sav'
        else:
            for key,value in filterr.items():
                filterr_file_name += "_"+str(key)+"_"+str(value)
            model_filename = path+str(resource_name)+'_'+str(metric_name)+filterr_file_name+'_MODEL.sav'
            upper_lower_filename = path+str(resource_name)+'_'+str(metric_name)+filterr_file_name+'_THRESHOLD.sav'
        joblib.dump(model, model_filename)
        joblib.dump({'upper':upper,'lower':lower},upper_lower_filename)
    else:
        return JSONResponse(content={'message':'Data is empty, So cannot train.'}, headers=headers)
    headers = {"Access-Control-Allow-Origin": "*"}
    return JSONResponse(content={'message':'Training done sucessfully'}, headers=headers)

@app.post("/predict")
def predict(item: fetch_data_item):
    anomaly_indicator = 'False'
    resource_name = item.resource_name
    metric_name = item.metric_name
    metric_value = float(item.metric_value)
    filterr = item.filterr
    filterr_file_name = ''
    headers = {"Access-Control-Allow-Origin": "*"}
    path = str(os.getenv('PARQUET_FILEPATH'))+'/model_univariate/'
    if filterr==None:
        model_filename = path+str(resource_name)+'_'+str(metric_name)+'_MODEL.sav'
        upper_lower_filename = path+str(resource_name)+'_'+str(metric_name)+'_THRESHOLD.sav'
    else:
        for key,value in filterr.items():
            filterr_file_name += "_"+str(key)+"_"+str(value)
        model_filename = path+str(resource_name)+'_'+str(metric_name)+filterr_file_name+'_MODEL.sav'
        upper_lower_filename = path+str(resource_name)+'_'+str(metric_name)+filterr_file_name+'_THRESHOLD.sav'
    try:
        model = joblib.load(model_filename)
        value_score = model.decision_function([[metric_value]])
        output = model.predict([[metric_value]])
    except Exception as e:
        return JSONResponse(content={'message':str(e)}, headers=headers)
    
    if output==1:
        msg = 'Not an Anomaly'
    else:
        anomaly_indicator = 'True'
        threshold = joblib.load(upper_lower_filename)
        if metric_value > threshold['upper'] or metric_value < threshold['lower']:
            msg = 'Global Anomaly'
        else:
            msg = 'Contextual Anomaly'
    
    return JSONResponse(content={'message':msg,'anomaly':anomaly_indicator}, headers=headers)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Threat Detection Univariate Model Isolation Forest",
        version="1.0",
        description="This API useful for detecting Anomaly using Isolation forest",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001, debug=True)