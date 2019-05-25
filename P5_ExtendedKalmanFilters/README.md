# Extended-Kalman-Filter
Udacity CarND Term 2, Project 1 - Extended Kalman Filters

INPUT: values provided by the simulator to the c++ program

["sensor_measurement"] => the measurement that the simulator observed (either lidar or radar)

OUTPUT: values provided by the c++ program to the simulator

["estimate_x"] <= kalman filter estimated position x

["estimate_y"] <= kalman filter estimated position y

["rmse_x"]

["rmse_y"]

["rmse_vx"]

["rmse_vy"]

## Build And Run
Run buildRun.sh script
and also you need to run client 'term2_sim' binary
They will communicate using uWebSocketIO.

## Output image
./result.png