nohup python train.py --cuda -d coco -v baseline -bk r18 1>1-coco-baseline-r18.txt 2>2.txt &

python train.py --cuda -d coco -v baseline -bk r18 -r ./weights/coco/baseline/baseline_33_0.12.pth