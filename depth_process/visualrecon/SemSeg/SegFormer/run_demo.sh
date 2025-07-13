IMAGE_FILE=demo/015344.jpg
CONFIG_FILE=local_configs/segformer/B3/segformer.b3.1024x1024.city.160k.py
CHECKPOINT_FILE=checkpoints/segformer.b3.1024x1024.city.160k.pth

python demo/image_demo.py \
    $IMAGE_FILE \
    $CONFIG_FILE \
    $CHECKPOINT_FILE \
    --device cuda:0 \
    --palette cityscapes