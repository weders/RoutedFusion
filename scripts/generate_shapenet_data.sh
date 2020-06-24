#!/bin/bash
# script to convert the Shapenet V1 dataset to the
# data format that is used in RoutedFusion

# path to shapenet core v1 dataset
SOURCE_PATH=$1
# path to save the processed data
DEST_PATH=$2

# path to the mesh-fusion tool
TOOL_PATH='deps/mesh-fusion'

mkdir -p $DEST_PATH

# read list=()
scenes=()

# reading train, val and test list
while read -r line; do
    reformatted=${line//$'\t'//}
    scenes+=("$reformatted")
done < lists/shapenet/routing/train.txt
while read -r line; do
    reformatted=${line//$'\t'//}
    scenes+=("$reformatted")
done < lists/shapenet/routing/val.txt
while read -r line; do
    reformatted=${line//$'\t'//}
    scenes+=("$reformatted")
done < lists/shapenet/routing/test.txt


# copying all scenes to destination
for s in "${scenes[@]}"; do
  cp -r $SOURCE_PATH/$s $DEST_PATH/$s

  ID_PATH="$DEST_PATH/$s"

  echo $ID_PATH

  mkdir -p "$ID_PATH/in"

  if [ -d "$ID_PATH/voxels" ]; then
      echo "$s already processed"
      continue
  fi

  echo "Converting meshes to OFF"
  echo "$ID_PATH/models/model_normalized.obj"
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
