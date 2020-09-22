#!/bin/bash
# script to convert the Shapenet V1 dataset to the
# data format that is used in RoutedFusion

# path to shapenet core v1 dataset
SOURCE_PATH=$1
# path to save the processed data
DEST_PATH=$2

CODE_PATH=$(pwd)

export PYTHONPATH=$CODE_PATH/deps/mesh-fusion/librender/:$PYTHONPATH
export PYTHONPATH=$CODE_PATH/deps/mesh-fusion/libfusiongpu/:$PYTHONPATH

# path to the mesh-fusion tool
TOOL_PATH='deps/mesh-fusion'

mkdir -p $DEST_PATH

# read list=()
scenes=()

# reading train, val and test list
while read -r line; do
    reformatted=${line//$'\t'//}
    scenes+=("$reformatted")
done < lists/shapenet/fusion/train.txt
while read -r line; do
    reformatted=${line//$'\t'//}
    scenes+=("$reformatted")
done < lists/shapenet/fusion/val.txt
while read -r line; do
    reformatted=${line//$'\t'//}
    scenes+=("$reformatted")
done < lists/shapenet/fusion/test.txt


# copying all scenes to destination
for s in "${scenes[@]}"; do

  # go back to project directory
  cd $CODE_PATH

  # parse category and object number
  cat="$(cut -d'/' -f1 <<< $s)"
  obj="$(cut -d'/' -f2 <<< $s)"

  # create destination path
  mkdir -p $DEST_PATH/$cat/$obj

  # copy object and corresponding material
  cp $SOURCE_PATH/$cat/$obj/model.obj $DEST_PATH/$cat/$obj/model.obj
  cp $SOURCE_PATH/$cat/$obj/model.mtl $DEST_PATH/$cat/$obj/model.mtl

  ID_PATH="$DEST_PATH/$s"

  echo $ID_PATH

  mkdir -p "$ID_PATH/in"

#  if [ -d "$ID_PATH/voxels" ]; then
#      echo "$s already processed"
#      continue
#  fi

  echo "Converting meshes to OFF"
  meshlabserver -i $ID_PATH/model.obj -o $ID_PATH/in/model.off;

  python "$TOOL_PATH/1_scale.py" --in_dir="$ID_PATH/in" --out_dir="$ID_PATH/scaled"
  python "$TOOL_PATH/2_fusion.py" --mode=render --in_dir="$ID_PATH/scaled" --depth_dir="$ID_PATH/depth" --out_dir="$ID_PATH/watertight"
  python "$TOOL_PATH/2_fusion.py" --mode=fuse --in_dir="$ID_PATH/scaled" --depth_dir="$ID_PATH/depth" --out_dir="$ID_PATH/watertight"
  python "$TOOL_PATH/3_simplify.py" --in_dir="$ID_PATH/watertight" --out_dir="$ID_PATH/out"

  mkdir -p "$ID_PATH/voxels"

  cd "$ID_PATH/out/"

  binvox -cb -bb -0.5 -0.5 -0.5 0.5 0.5 0.5 -rotz -rotz -rotz -rotx -pb -d 128 *

  fname=(*.binvox)
  basename "$fname"
  f="$(basename -- $fname)"
  y=${f%.off}
  filename=${y##*/}.128.binvox
  mv $fname ../voxels/$filename

done