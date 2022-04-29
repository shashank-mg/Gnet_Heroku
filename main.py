from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


@app.get("/ping")
async def ping():
    return "Hello"

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL = tf.keras.models.load_model('model2.h5')

CLASS_NAMES = ['Coffee_Healthy',
               'Cotton_healthy',
               'Cotton_leaf_blight',
               'Maize_Blight',
               'Maize_Common_Rust',
               'Maize_Gray_Leaf_Spot',
               'Maize_Healthy',
               'Rice_Brown_spot',
               'Rice_Healthy',
               'Rice_Hispa',
               'Rice_LeafBlast',
               'Sugarcane_Bacterial_Blight',
               'Sugarcane_Healthy',
               'Sugarcane_Red_Rot',
               'coffee_miner',
               'coffee_rust']


def read_file_as_image(img):
    image = np.array(Image.open(BytesIO(img)))
    # print(image)
    return image


@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):

    bytes = await file.read()
    image = read_file_as_image(bytes)
    img_batch = np.expand_dims(image, 0)
    # print(np.expand_dims(image, 0))
    prediction = MODEL.predict(img_batch)
    # print(CLASS_NAMES[np.argmax(prediction[0])])
    # print("confidence ", round(np.max(prediction[0])*100))
    return {CLASS_NAMES[np.argmax(prediction[0])], round(np.max(prediction[0])*100)}

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)
