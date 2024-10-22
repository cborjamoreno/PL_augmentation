comando='slic_cli' # superpixel method

data=$1/images
data_train=$data/train
data_test=$data/test
pathname=$data

# name of the dataset
name="$(echo $(basename $(dirname $data)))"

# output paths
out_test=ML_Superpixels/Datasets/$name/superpixels/test
out_train=ML_Superpixels/Datasets/$name/superpixels/train

mkdir -p $out_train
mkdir -p $out_test

shift

#For each value (N), creates segmentations of N segments
for i in $@; do
 	ML_Superpixels/bin/$comando --input $data_train --output $out_train/superpixels_$i --contour --csv  --superpixels $i > /dev/null 2>&1
 	# ML_Superpixels/bin/$comando --input $data_test --output $out_test/superpixels_$i --contour --csv  --superpixels   $i
 done

