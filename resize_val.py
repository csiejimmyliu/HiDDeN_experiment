import torchvision.transforms as T
import torchvision
import os

save_path=os.path.join('./resize_coco_val')

if not os.path.exists(save_path):
    os.makedirs(save_path)


resize_transform = T.Compose([
    T.Resize((512,512))
])


dataset=torchvision.datasets.ImageFolder('./dataset/val/',resize_transform)
for i in range(len(dataset)):
    dataset[i][0].save(os.path.join(save_path,f'{i}.png'))
    