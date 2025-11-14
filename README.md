# Mount and copy the dataset
```
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

!cp -r "/content/drive/MyDrive/BRaiN/Combined Dataset/train" /content/
!cp -r "/content/drive/MyDrive/BRaiN/Combined Dataset/test" /content/
print("Data copied!")
