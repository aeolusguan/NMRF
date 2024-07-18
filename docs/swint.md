Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1noY4qOR4K9_Eiu7FK0bz4M2bG_WUxmMA?usp=sharing). For submission to KITTI 2012 and 2015 online test sets, you can run:
```shell
python inference.py --dataset-name kitti_2015 --output $your_directory --config-file configs/kitti-mix-train-swint.yaml SOLVER.RESUME pretrained/kitti_swint.pth
```
and
```shell
python inference.py --dataset-name kitti_2012 --output $your_directory --config-file configs/kitti-mix-train-swint.yaml SOLVER.RESUME pretrained/kitti_swint.pth
```