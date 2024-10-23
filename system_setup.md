Imagenet Dataset:
1. Download datasets ( ~/data/imagenet and ~/data/stl10_binary) on one node by (imagenet link: train set https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar, val set https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar) and (stl10) running the training script with world_size=1 to let pytorch download.
   1. use the following bash code to extract ILSVRC2012_img_train.tar
   ```
    mkdir train val
    mv ILSVRC2012_img_train.tar train
    mv ILSVRC2012_img_val.tar val
    tar -xvf ILSVRC2012_img_train.tar
    for fn in n*.tar
    do
        d=($(echo $fn | tr "." "\n"))
        mkdir $d
        tar -xvf $fn --directory $d
    done
    tar -xvf ILSVRC2012_img_val.tar
   ```
   2. Run `python pcode/tools/imagenet_val_split.py --imagenet_dir ~/data/imagenet --imagenet_val_index pcode/tools/ImageNet_val_labels.txt` to split val into labelled directories
   2. The folder structure of imagenet should looks like this:
    ```
    ~/data
    └── imagenet
        ├── train
        │   ├── n01440764
        │   │   ├── ...
        │   │   └── ...JPEG
        │   ├── n01443537
        │   └── ...
        └── val
            ├── n01440764
            │   ├── ...
            │   └── ...JPEG
            ├── n01443537
            └── ...
    ```
    3. run `cd ~ && find data/imagenet/train -type f | wc -l` to check if number of samples is correct: should show 1281168
    4. run `cd ~ && find data/imagenet/val -type f | wc -l` to check if number of samples is correct: should show 50000
    5. Use https://github.com/xunge/pytorch_lmdb_imagenet to convert dataset into lmdb.

