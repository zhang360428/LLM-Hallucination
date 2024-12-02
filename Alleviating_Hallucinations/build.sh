#!/bin/bash
#约定的镜像tag
image_name='ai_cq2:lastest'
#约定的docker 镜像文件
image_tar_name='image.tar'
#选手自己的dockerfile目录
dockerfile_path='./dockerfile'
docker build -f $dockerfile_path -t $image_name .
docker save -o  $image_tar_name $image_name
#打包答案压缩文件，最后上传docker-cq2.zip即可
zip -r docker-cq2.zip dockerfile image.tar source build.sh
