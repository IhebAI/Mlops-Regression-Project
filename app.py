
import os
from fastapi import FastAPI, HTTPException

from RegressionProject.pipeline.predict_pipeline import CustomData, PredictPipeline
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the prediction API. Use the /predict endpoint to get predictions."}

@app.post("/predict")
async def predict(custom_data: CustomData):
    try:
        data_df = custom_data.get_data_as_data_frame()
        pipeline = PredictPipeline()
        predictions = pipeline.predict(data_df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train():
    os.system("python main.py")
    return "Training done successfully!"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
