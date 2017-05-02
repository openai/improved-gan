export word=imagenet
export pixels=128
echo $HOSTNAME >> ${word}.record.txt
CUDA_VISIBLE_DEVICES=$GPUS python train_${word}.py --dataset imagenet_train --is_train True --checkpoint_dir gan/checkpoint_${word} --image_size ${pixels} --is_crop True --sample_dir gan/samples_${word} --image_width ${pixels} --batch_size 16
