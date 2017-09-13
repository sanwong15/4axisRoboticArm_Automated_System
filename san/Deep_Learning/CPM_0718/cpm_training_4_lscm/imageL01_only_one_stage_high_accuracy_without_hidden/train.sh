postfix=`date +"%m_%d_%y"`

#/DATA/xzhu/Robotics/cpm/caffe/build/tools/caffe train \
#-solver ./pose_solver.prototxt \
#-weights ./prototxt/shortPin_longPin_all/caffemodel/pose_iter_27000.caffemodel \
#-gpu 4,5,6,7  2>&1 | tee ./prototxt/20160601/caffemodel/train_$postfix.log


#nohup /DATA/xzhu/Robotics/cpm/caffe/build/tools/caffe train \
#-solver ./pose_solver.prototxt \
#-gpu 4,5,6,7  2>&1 | tee ./train_$postfix.log &



#nohup /DATA/xzhu/Robotics/cpm/caffe/build/tools/caffe train \
#-solver ./pose_solver.prototxt \
#-snapshot ./capacitor_20160601_iter_78000.solverstate \
#-gpu 4,5,6,7  2>&1 | tee ./train_from78000_$postfix.log &


#nohup /DATA/xzhu/Robotics/cpm/caffe/build/tools/caffe train \
#-solver ./pose_solver.prototxt \
#-snapshot ./capacitor_20160601_iter_373000.solverstate \
#-gpu 4,5,6,7  2>&1 | tee ./train_from373000_$postfix.log &


#nohup /DATA/xzhu/Robotics/cpm/caffe/build/tools/caffe train \
#-solver ./pose_solver.prototxt \
#-snapshot ./capacitor_20160601_iter_478000.solverstate \
#-gpu 4,5,6,7  2>&1 | tee ./train_from478000_$postfix.log &


nohup /DATA/xzhu/Robotics/cpm/caffe/build/tools/caffe train \
-solver ./pose_solver.prototxt \
-gpu 4,5,6,7  2>&1 | tee ./train_$postfix.log &


#nohup /DATA/xzhu/Robotics/cpm/caffe/build/tools/caffe train \
#-solver ./pose_solver.prototxt \
#-snapshot ./capacitor_20160601_imageL01_iter_340000.solverstate \
#-gpu 4,5,6,7  2>&1 | tee ./train_from340k_$postfix.log &







