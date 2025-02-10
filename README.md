# AMS

## Requirements
Use Python 3.9

## Installation

Detailed instruction are also available in the (printed) TIAGo handbook (in the lab).

### Tiago

#### Pal Ubuntu

To install the Tiago Operating System first open the service panel at the front bottom right if the robot.
Plug in a HDMI Display and a USB Keyboard and insert a USB Stick with Pal Ubuntu Image.
Then start Tiago and press F2 repeatedly until the BIOS menu opens.
Select the USB Stick with the Pal Ubuntu Image as first Boot Option.
And leave the BIOS with the save and exit option.
You will have three options for installation.
Select the ```Install TIAGo``` option and then configure country and keyboard layout.
After the Installation shutdown the robot and reset the internal disk as first Boot Option in the BIOS.

#### Network Configuration

To connect TIAGo to a WiFi network after installation,
connect to the Tiago via its WiFi Hotspot or an Ethernet cable
The password can be found the Tiago handbook.
Then open the Web Commander, select the Networking tab and enter the configuration.
Enter the SSID and Password of the WiFi Network you want to connect to enable DHCP or configure the address manually.

### Development

#### Pal Ubuntu

To install the Development Version of the Pal Ubuntu Image perform the same steps as in
[Installation Tiago](#pal-ubuntu) but select the ```Install Development TIAGo``` option.
Alternatively you can install the Pal Ubuntu Image in a Virtual Machine.
**Be careful if you want to install the Image on a real machine directly, because you can not choose the target disk or partition.** 

#### Git Repository
Clone the ```AMS``` git repository

```bash
git clone https://github.com/lufla/AMS.git
```

#### Env Config
Copy the files ```example.env``` and ```example.env.json``` from the directory ```pcb_detection/``` into to root directory of the git repository ```AMS/``` and rename them to ```.env``` and ```.env.json```.

Adjust the camera parameters in the ```.env``` file if necessary.

#### Python Dependencies

To install the needed python dependencies execute the following command.
```bash
pip install -r pcb_detection/requirements.txt
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
To stop recording press ```q``` in the image window.

<<<<<<< HEAD
The images will be saved in ```PCB_Detection_2/calibration/```.
=======
The images will be saved in ```pcb_detection/calibration/tiago/```.
>>>>>>> cf10c9c7c64a7632d2293eeab01f79391590dbcc

For calibrating the webcam you can use the ```take_calibration_images.py``` python script.

To calculate and save the calibration values execute the ```calibrate_camera`.py``` script.
To select the camera for which the camera matrix should be calculated, set the ```CAMERA``` variable to one of the following values.

```python
CAMERA = CAMERA_WEBCAM
CAMERA = CAMERA_HEAD
CAMERA = CAMERA_GRIPPER
```

The calibration script will display the image and highlight the detected key points.
After the calibration is done it will output the mean reprojection error.

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

##### PCB and IC Detection

The ```main.py``` script only includes the PCB detection and works with a webcam.

After starting the script, multiple images will be displayed.

```reference```: shows the reference image of the PCB

```frame```: displays the camera input together with an overlay of the detected contours, the gerber overlay and component positions of the PCB

```canny```: displays the canny filtered camera input

```thresh```: displays the thresholded camera input used for contour detection

```perspective```: shows the current camera input corrected by the inverse perspective transformation to show the detected PCB, together with an overlay of the gerber image and the component positions

```reference_cutout```: shows a cutout from the reference image around the component ```IC1```.

```input_cutout```: shows the camera input together with the region where the ```reference_cutout``` image was matches as a template

---

The ```main_detection.py``` script can connect to ROS and use the head or gripper camera of Tiago.
It performs PCB Detection as well as IC Detection.
To switch between Webcam and Tiago Cameras set to ```USE_WEBCAM``` variable to either ```True``` or ```False```
The script displays the additional image ```All Rotations``` in which IC labels and the corresponding ICs next to the label are detected and highlighted.


##### Robot Control

The scripts ```roslibpy-arm-test.py``` and ```roslibpy-head-test.py``` can be used to control the arm and head of tiago manually.

The head has two joints to control.
Head joint 1 pans left and right and head joint 2 tilts up and down.

The arm can be controlloed with x and y coordinates relative to the first arm joint. The x axis runs back to front and the y axis runs from right to left, both from the view of the robots head.
Additionally the gripper rotation has to be given.

All angles are measured in radians and distances are measured in meters.

The script ```main_manual.py``` can perform multiple commands.

```end```: Stops the script

```hello```: Prints "hello"

```detect_cutout```: Uses template matching to find the pcb cutout around the currently selected component and displays the result. Can be stopped by pressing ```q```

```detect_pcb```: Detects the PCB Outline and based on this, calculates its position and displays the results. Can be stopped by pressing ```q```. The last PCB position gets stored for further use. The terminal outputs the 3D coordinates 

```calculate_component_pcb_position```: Calculates the position of the currently selected Component on the PCB in relation to the camera. The result is stored and displayed in the terminal

```end```: stops the script

```end```: stops the script


    "detect_cutout": detect_cutout,
    "detect_component": detect_component,
    "detect_pcb": detect_pcb,
    "calculate_component_pcb_position": calculate_component_pcb_position,
    "set_component_index": set_component_index,
    "move_arm_over_component": move_arm_over_component,
    "move_head": move_head,
    "reset_arm": reset_arm,
    "show_head_transform": show_head_transform,
