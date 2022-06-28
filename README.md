# HiDDeN_experiment
Extended experiment of HiDDeN model to enhance robustness based on the code from [ando-khachatryan](https://github.com/ando-khachatryan/HiDDeN)

# Data

Using 10,000 images for training and 1,000 images for validation. Following the original paper,choosing those 10,000 + 1,000 images randomly from one of the coco datasets.

The data directory has the following structure:

```bash
<data_root>/
  train/
    train_class/
      train_image1.jpg
      train_image2.jpg
      ...
  val/
    val_class/
      val_image1.jpg
      val_image2.jpg
      ...
```

# Running
cd to the specific HiDDeN folder.

Following the instruction in this [Github](https://github.com/ando-khachatryan/HiDDeN)