import torch
from torchvision import transforms
from torchvision.models.inception import inception_v3
# from metric.inception_3 import inception_v3
from PIL import Image


# # 测试图像的预测类别
f = open('labels.txt', mode='r')
class_names = f.readlines()

model = inception_v3(pretrained=True, transform_input=False)
model.eval()

transform = transforms.Compose([transforms.Resize((299, 299)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

img = Image.open("dog.jpg") 
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)  # [bs, c, h, w]

# move the input and model to GPU for speed if available
device = torch.device("cuda:0")
if torch.cuda.is_available():
    batch_t = batch_t.to(device)
    model.to(device)

with torch.no_grad():
    output = model(batch_t)  # 是softmax激活前的输入，即logits. [bs, 1000]

percentage = torch.nn.functional.softmax(output[0], dim=0)
_, indices = torch.sort(output, descending=True)
top_5 = [(class_names[idx], percentage[idx].item()) for idx in indices[0][:5]]
print(top_5)

