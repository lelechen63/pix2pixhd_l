### Using labels only
#python train.py --name label2city_512p --dataset densepose --batchSize 16

python train.py --name syn512p --dataroot /data/syn --model noback --dataset syn --batchSize 16 --serial_batches --gpu_ids 1,2,3,4