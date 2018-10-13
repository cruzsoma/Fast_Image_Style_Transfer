train_image_path = "train2014"
style_image = "images/mosaic.jpg"
style_name = "mosaic"
model_path = "models"
loss_model = "vgg_16"
loss_model_file = "pretrained/vgg_16.ckpt"
checkpoint_exclude_scopes = "vgg_16/fc"

content_layers = "vgg_16/conv3/conv3_3"
style_layers = ["vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2", "vgg_16/conv3/conv3_3", "vgg_16/conv4/conv4_3"]

content_weight = 1.0
style_weight = 100.0
tv_weight = 0.0

image_size = 256
batch_size = 4
epoch = 2