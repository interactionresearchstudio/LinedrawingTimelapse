# LinedrawingTimelapse
A time lapse that uses Canny edge detection to produce a line drawing effect. For the ProbeTools PiCamera.

## Requirements
- OpenCV 3.0+
- Python 3
- imutils
- numpy
- CameraController submodule

## Installation

###Raspberry Pi
In order to install all the prerequisites on the Raspberry Pi, please use the 
[VisionCam install scripts](https://github.com/interactionresearchstudio/VisionCamInstallScripts).
All scripts, automatic Xsession starting, OpenCV compiling is taken care of for you. 

### Mac / Linux / Windows
Install the above requirements, pull in the CameraController submodule, and the script should be able to run 
without Raspberry Pi specific libraries. Use the keys 1-4 in substitution of the GPIO buttons found on the VisionCam. 

## Contributions
Fork this repository and make sure you test your code against the `dev` branch.

## Bug reports
Please be as specific as possible when reporting bugs, including which system you are using, specific hardware, and
logging files. 
