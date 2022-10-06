#!/bin/sh

wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat ./faceutils/dlibutils/

pip install gdown
gdown https://drive.google.com/uc?id=1c8GVwfWg7QEZUaKSKb6z-24jXt9vRuGd
mkdir ckpts
mv sow_pyramid_a5_e3d2_remapped.pth ./ckpts/

gdown https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812
mv 79999_iter.pth ./faceutils/mask/resnet.pth
