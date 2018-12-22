# ai2r

To train:
`python ai2r.py -d dataset_name -m inceptionv3_pretrained --gpu 1`

`dataset_name` is the name of the dataset which has to be put in raw directory.
For example: python ai2r.py -d aircrafts -m inceptionv3_pretrained --gpu 1

To predict:
`python predict.py -i path_to_image`
