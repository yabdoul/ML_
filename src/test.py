from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torch
from model import myModel

model = myModel()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()


image = Image.open("image.png").convert("L") 

image = ImageOps.invert(image)


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
image = transform(image).unsqueeze(0)  


with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print(f"Predicted digit: {predicted.item()}")
