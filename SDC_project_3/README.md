# End to End Learning for Self-Driving Car
 
picture  
 
## Introduction
 
This is Udacity's Self-Driving Car Nanodegree Project.
Our goal is self-driving by using behavior cloning. It means the process is to mimic driving behavior of driver without any detection such as lane marking detection, path planning. In this project, we train our car by using recorded driving images in a simulator provided by Udacity. In the simulator, we record images, steering angle, throttle, speed, etc. The main data is image and steering angle. After training our CNN, the network model predict a steering angle in every frames, so our trained car can drive autonomously. Also, we test the network model with untrained track.  
 
Before start this project,  
 
* NVIDIA 2016 Paper [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
would be good guide to understand the exact concept.  
* In a MIT winter 2017 course [Deep Learning for Self-Driving Cars](http://selfdrivingcars.mit.edu/deeptesla/), they are working on a **deeptesla** project and similar with this project.  
* [comma.ai](https://github.com/commaai/research) shared their dataset & codes. 
 
 
## Installation & Environment
 
### Installation
 
* Simulator  
 
    [Windows 64 bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)  
    [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)  
    [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)  
 
 
* Dataset  
[data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) (Track1 data provided by Udacity)
 
 
### Environment
  
#### software  
  
    Windows x64, tensorflow 0.12.1, keras, Python 3.5, OpenCV 3.1.0  
 
#### hardware  
  
    CPU : i7-4720HQ 2.60Hz, GPU : GTX 970M, Memory : 16GB  
    
## Files
  
`model.py` : training model  
`drive.py` : drive a car in a simulator  
`model.json` : saved training model  
`model.h5` : saved training weight  
  
To test self-driving(autonomous mode), type `python drive.py model.json` on terminal.  
 
## Data collection
 
picture  
 
 
In the simulator, we have two modes and two racing tracks.  
 
`TRAINING MODE` : Coltrol a car with keyboard or joystick. Collect image datas by clicking a record button.  
 
`AUTOMOMOUS MODE` : Test my trained car.  
 
**Data collection should only be performed in Track1.**  
We'll check our network model operates well in an untrained track.(Track2)  
 
In an autonomous mode, The trained car'll drive by mimicing our control. That means that if we want high-quality results, we need cautious vehicle-control. So we could use joystick instead of keyboard to get soft angle change.  
In udacity course, they said **"garbage in, garbage out."**  
 
Also when we record driving, we need data about 'steer back to the middle from the side of the road'. If all of our data is collected with driving in the middle of the road, our car couldn't recover to the middle from the side. But it's really hard to collect many recovery data because our dataset must not have 'weaving out to the side' data. That will exacerbate the accuracy.  
So, that's why we have 3 cameras. We could map recovery paths from each camera. If middle camera image is pointing 'left turn', we could get 'soft left turn' with left camera and 'sharp left turn' with right camera.   
 
I used [udacity dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) because I didn't have a joystick. I also have my dataset using keyboard but using udacity dataset is easy to compare the result with other students' result.  
 
![alt text][picture1]
 
 
 
 
[picture1]: /images/data_distribution.png "angle_distribution"
 
 
 
 
 
 
 
 
 
 
 
 
2. data preprocessing
used python generator
 
using Left, Right Camera
 
3. model architecture
 
result
 
- visualization
    interpolation, size 1:2
    visualize all in one image(layer1, layer2)
- video
 
reflection
 
Really impressive project. I have looked forward to start this project since I check the curriculum of Udacity's SDC ND.
from collecting data to design a network architecture, every parts were important. During completing this project, I could use
keras and learn generator. I extreamly realized the importance of data quality. 
 
```diff+ this will be highlighted in green
- this will be highlighted in red
```
 
```json
   // code for coloring
```
```html
   // code for coloring
```
```js
   // code for coloring
```
```css
   // code for coloring
```
// etc.
 
Some Markdown text with <span style="color:blue">some *blue* text</span>.