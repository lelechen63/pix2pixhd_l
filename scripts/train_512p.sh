### Using labels only
# python train.py --name label2city_512p --dataset densepose --batchSize 20 --gpu_ids 5,7

python train.py --name syn512p --dataroot /data/syn --model noback --dataset syn --batchSize 32  --gpu_ids 1,2,3,4