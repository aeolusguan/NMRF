Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1noY4qOR4K9_Eiu7FK0bz4M2bG_WUxmMA?usp=sharing). For submission to KITTI 2012 and 2015 online test sets, you can run:
```shell
python inference.py --dataset-name kitti_2015 --output $your_directory --config-file configs/kitti-mix-train-swint.yaml SOLVER.RESUME pretrained/kitti_swint.pth
```
and
```shell
python inference.py --dataset-name kitti_2012 --output $your_directory --config-file configs/kitti-mix-train-swint.yaml SOLVER.RESUME pretrained/kitti_swint.pth
```

To train on SceneFlow dataset, you can run following commands in the project directory:
```shell
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
python main.py --num-gpus 4 --config-file configs/sceneflow_swint.yaml BACKBONE.WEIGHT_URL swin_tiny_patch4_window7_224.pth
```
The pretrained weight on SceneFlow is available at [google drive](https://drive.google.com/file/d/1eR2ARJYqMyGmInIIQ7XV3RvAwYlvDrBV/view?usp=drive_link).
