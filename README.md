# Mount and copy the dataset
```
from google.colab import drive
drive.mount('/content/drive',force_remount=True)

# CORRECT PATH: Only ONE "BRaiN"
!cp -r "/content/drive/MyDrive/Colab Notebooks/Combined Dataset/train" /content/
!cp -r "/content/drive/MyDrive/Colab Notebooks/Combined Dataset/test" /content/

print("Data copied to Colab!")
!ls /content/train | head -5
```

# Defines a custom PyTorch Dataset class (MRIDataset) used to load MRI images for training and testing the EfficientNet model.
```
from torch.utils.data import Dataset, DataLoader
 from PIL import Image
 import torch
 from torchvision import transforms
 import os
 class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        # Auto-detect classes (alphabetical)
        class_names = sorted([d for d in os.listdir(root_dir) if 
os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {name: i for i, name in 
enumerate(class_names)}
        print("CLASS MAPPING:")
        for k, v in self.class_to_idx.items():
            print(f"  {k} → {v}")
        for class_name in class_names:
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', 
'.jpeg')):
                    self.images.append(os.path.join(class_path, 
img_name))
                    self.labels.append(self.class_to_idx[class_name])
        print(f"Loaded {len(self.images)} images from {root_dir}")
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)
```
# Prepares the data for training and testing the model.

```
 transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ])
 train_dataset = MRIDataset('/content/train', transform=transform)
 test_dataset = MRIDataset('/content/test', transform=transform)
 train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
num_workers=2)
 test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 print("\nDATA LOADERS READY!")
```
# Prepares the EfficientNet-B0 model for training.
```
 !pip install timm -q
 import timm
 import torch.nn as nn
 import torch.optim as optim
 # Load EfficientNet-B0 (pretrained on ImageNet)
 model = timm.create_model('efficientnet_b0', pretrained=True, 
num_classes=4)
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
 criterion = nn.CrossEntropyLoss()
 optimizer = optim.Adam(model.parameters(), lr=0.0003)  # Lower LR for 
fine-tuning
 print("EfficientNet-B0 loaded! Ready for training.")
```
# Evaluates the trained EfficientNet-B0 model on the test dataset and calculates final accuracy
```
 model.eval()
 correct = total = 0
 with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
 accuracy = 100 * correct / total
 print(f"\n🎉 FINAL ACCURACY: {accuracy:.2f}%")
 print(f"Correct: {correct} / {total}"
```
# (Optional, but recomended for easiness)
```
torch.save(model.state_dict(), '/content/efficientnet_b0_final.pth')
 !mkdir -p "/content/drive/MyDrive/Alzheimer_MVP"
 !cp /content/efficientnet_b0_final.pth 
"/content/drive/MyDrive/Alzheimer_MVP/"
 print("🎉 MODEL SAVED PERMANENTLY!")
```
# Creates a Streamlit web app to run your trained "EfficientNet-B0" Model,
and then uses Ngrok to host the app online so you get a public link.
(login to ngrok , get your token and paste in the cell)

```
 %%writefile /content/app.py
 import streamlit as st
 import torch
 from PIL import Image
 from torchvision import transforms
 import timm
 @st.cache_resource
 def load_model():
    model = timm.create_model('efficientnet_b0', pretrained=False, 
num_classes=4)
    
model.load_state_dict(torch.load('/content/efficientnet_b0_final.pth', 
map_location='cpu'))
    model.eval()
    return model
 model = load_model()
 transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ])
 classes = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 
'Very Mild Impairment']
 st.title("🎉 Alzheimer MRI AI")
 st.markdown("**EfficientNet-B0 • 98%+ Accuracy • 5.3M params**")
 uploaded = st.file_uploader("Upload MRI", type=['jpg', 'jpeg', 'png'])
 if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="MRI Scan", width=350)
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = torch.softmax(model(x), 1)[0]
        i = pred.argmax().item()
    st.success(f"**Diagnosis: {classes[i]}**")
    st.bar_chart({c: float(pred[j]) for j, c in enumerate(classes)})
 Overwriting /content/app.py
 # 1. INSTALL pyngrok + streamlit
 !pip install pyngrok streamlit -q
 # 2. IMPORT & SET TOKEN
 from pyngrok import ngrok
 import time
 (get from: 
https://dashboard.ngrok.com/get-started/your-authtoken)
 ngrok.set_auth_token("PASTE YOUR NGROK TOKEN HERE") 
 # 3. KILL OLD PROCESSES
 !pkill -f streamlit
 !pkill -f ngrok
 # 4. START STREAMLIT APP
 !nohup streamlit run /content/app.py --server.port=8501 -
server.headless=true > log.txt 2>&1 &
 # 5. WAIT & LAUNCH NGROK
 time.sleep(15)
 tunnel = ngrok.connect(8501, bind_tls=True)
 # 6. PRINT LIVE LINK
 print(f"\nLIVE DEMO READY!")
 print(f"Click here: {tunnel.public_url}"
 ```
