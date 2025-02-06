# AMS

## Requirements
Use Python 3.9

## Installation

### Tiago

#### Pal Ubuntu

### Development

#### Pal Ubuntu

#### Git Repository
Clone the ```AMS``` git repository

```bash
git clone https://github.com/lufla/AMS.git
```

#### Env Config
Copy the files ```example.env``` and ```example.env.json``` from the directory ```PCB_Detection_2/``` into to root directory of the git repository ```AMS/``` and rename them to ```.env``` and ```.env.json```.

Adjust the camera parameters in the ```.env``` file if necessary.

#### Python Dependencies

To install the needed python dependencies execute the following command.
```bash
pip install -r PCB_Detection_2/requirements.txt
``` 


#### ROS Packages

First perform the steps from [ROS Connection](#ros-connection)

If the ros package was not built yet or deployed to tiago yet, execute the following commands.

```bash
catkin_make
rosrun pal_deploy deploy.py -p my_controller_pkg tiago-158c
```

## Setup

### Tiago

Before using Tiago check if time matches the time of development pc using the ```date``` command.

To set the time on tiago you can use the following command. Replace ```00:00 CET``` with the time of the development pc.

```bash
ssh pal@tiago-158c
date --set="00:00 CET"
```

### Development

#### Calibration

Calibration images for the tiago cameras are already included in the git repository.
Additional images can be taken with the ```ros-subscriber-images.py``` python script.
Select the camera by setting the ```CAMERA``` variable to ```HEAD``` or ```GRIPPER``` and the ```SAVE_IMAGES``` variable to ```True```.

```python
CAMERA = HEAD
CAMERA = GRIPPER
SAVE_IMAGES = True
```

The script will take an image every 2 seconds.
To stop recording press ```q```.

The images will be saved in ```PCB_Detection_2/calibration/tiago/```.

For calibrating the webcam you can use the ```take_calibration_images.py``` python script.

To calculate and save the calibration values execute the ```calibrate_camera`.py``` script.
To select the camera for which the camera matrix should be calculated, set the ```CAMERA``` variable to one of the following values.

```python
CAMERA = CAMERA_WEBCAM
CAMERA = CAMERA_HEAD
CAMERA = CAMERA_GRIPPER
```

## Usage

### Tiago Start

After every restart of tiago.
Replace ```0.0.0.0``` with ip address of the development pc
```bash
ssh pal@tiago-158c
sudo addLocalDns -u development -i 0.0.0.0
```

Before using the python scripts also run the following commands.
```bash
rosnode kill /pal_head_manager
rosrun tiago_bringup end_effector_camera.sh & disown
```


### Development

#### ROS Connection

Connect the development pc to ROS on Tiago.

```bash
export ROS_MASTER_URI=http://tiago-158c:11311
export ROS_IP=$(hostname -I)
```

To use the ROS Packages you need to change to the ros workspace directory and source the setup script.

```bash
cd ctrl2_ws
source devel/setup.bash
```

#### ROS Controllers

First perform the steps from [ROS Connection](#ros-connection)

To bring the arm to a horizontal position run thk_init_pub.

```bash
rosrun my_controller_pkg thk_init_pub
```

For the python scripts to control the tiago arm, start the thk_arm_xya2 controller with this command.

```bash
rosrun my_controller_pkg thk_arm_xya2
```

#### Python Scripts

##### PCB Detection

##### IC Detection