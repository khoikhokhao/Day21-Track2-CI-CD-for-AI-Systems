from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

# Đọc tên bucket từ biến môi trường
CLOUD_BUCKET = os.environ.get("CLOUD_BUCKET", "default-bucket")
MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

def download_model():
    """Tải file model.pkl từ AWS S3 về máy khi server khởi động."""
    try:
        s3 = boto3.client('s3')
        s3.download_file(CLOUD_BUCKET, MODEL_KEY, MODEL_PATH)
        print("Model đã được tải xuống từ AWS S3.")
    except Exception as e:
        # Lần đầu tiên chạy, model chưa có trên S3 nên sẽ bắt lỗi này
        print(f"Chưa có model trên S3 (sẽ có sau khi chạy CI/CD): {e}")

download_model()

# Kiểm tra xem file model đã tồn tại chưa rồi mới load
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe server."""
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    """Endpoint suy luận."""
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    pred = model.predict([req.features])[0]
    
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": int(pred), "label": labels[int(pred)]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)