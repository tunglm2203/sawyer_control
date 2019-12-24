#!/usr/bin/env bash

rosservice call gazebo/delete_model '{model_name: cylinder}'
#rosrun gazebo_ros spawn_model -file /home/tung/ros_ws/src/sawyer_aim/models/cylinder/model.sdf -sdf -x 0.75 -y -0.1 -z 0.9 -model cylinder
rosrun gazebo_ros spawn_model -file /home/tung/ros_ws/src/sawyer_aim/models/cylinder/model.sdf -sdf -x $1 -y $2 -z $3 -model cylinder
