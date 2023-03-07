import cv2 as cv
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
import torch.nn.functional as F
import torchvision.transforms as transforms
from model import CNNModel
import __main__

app = FastAPI()

@app.post("/")
async def preds(file: UploadFile = File(...)):
    setattr(__main__, "CNNModel", CNNModel)
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(open('../cnn_model.pth', 'rb'))
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    f = cv.imdecode(nparr, cv.IMREAD_COLOR)
    # transform image input
    transform_norm = transforms.Compose([transforms.ToTensor()])
    # get normalized image
    img_normalized = transform_norm(f).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)
    with torch.no_grad():
        model.eval()  
        output = model(img_normalized)
        index = output.data.cpu().numpy().argmax()
        classes = ['Broccoli', 'Cabbage', 'Cucumber']
        class_name = classes[index]
        scores = F.softmax(output, dim=1)
        max_score= torch.max(scores)

        return {
            "probability": max_score.item(),
            "result": class_name,
        }
