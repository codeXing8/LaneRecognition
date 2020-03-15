## Algorithms for automotive lane markings recognition
### Deployment and sample data
#### Environment
It is recommended to use a virtual Python environment - created with e. g. Conda.
```bash
chmod +x install_dependencies.sh && ./install_dependencies.sh
```

### List of methods
#### 1D-Line fitting to edge detection results under triangular mask
<p align="center">
    <img src="LaneRecognition001.gif" alt="Exemplary lane recognition algorithm results 001"/>
</p>


Inside one frame (image) on a video stream, lanes are detected via edges.
Thus, this algorithm depends on visible lane markings.
By discarding short edges and averaging all kept edge positions, a 1D-polynomial fit (line) is used to determine slope and position of a single, averaged edge.


Based on both line parameters, the edge's length is increased from the lower horizontal image border to the image's vanishing point. To exclude long but irrelevant edges surrounding the actual lane, the area between the lower image boundary and the vanishing points is masked.


A triangular mask excludes everything outside this region of interest (ROI).
Inside the ROI, the approach is applied as described above. Furthermore, general filter techniques like gradient limitation and sliding average considering the line parameters where used.


#### Warped image edge detection and windowed 2D-polynomial fit
<p align="center">
    <img src="LaneRecognition002.gif" alt="Exemplary lane recognition algorithm results 002"/>
</p>

Inspired by [Ross Kippenbrock](https://www.youtube.com/watch?v=VyLihutdsPk).

As well as in the first method, Canny edge detection is applied to the image.
In addition, a color filter is applied to the results. Thus, only bright image parts yielding edges are taken into consideration.
The edges are represented by gray or white pixels while any other image parts are black.


The resulting image is warped (a perspective transformation is performed) and hence, the lane can be regarded from bird's-eye-view within the original image boundaries. By this method, bright lane markings are now aligned vertically along the image. This algorithm does not work without visible lane markings.


Along the image width, non-black image points are counted vertically resulting in a brightness histogram.
The maximum of the histogram values in each image half is, in theory, the center of a lane marking.
Rectangular masks are then applied onto the image on both sides, left and right, to reduce the image to the lane marking areas.
Since this algorithm was designed with curved roads in mind, the rectangular mask is vertically separated into so-called windows.
Each individual window is shifted to the current lane marking center along the image height for higher accuracy.


In each window, bright image points are considered and their coordinates are stored.
As a result, a set of point coordinates in x and y for each lane marking, left and right, can be used to fit a 2D-polynomial
along the lane marking. Between the fitted lane data plots, a polygon is spanned to denote the lane.
Finally, the perspective transformation is inverted again to recover the actual traffic scene.


#### Post-processed convolutional neural network with Encoder-Decoder architecture
<p align="center">
    <img src="LaneRecognition003.gif" alt="Exemplary lane recognition algorithm results 003"/>
</p>

Inspired by [Michael Virgo](https://towardsdatascience.com/lane-detection-with-deep-learning-part-2-3ba559b5c5af).

Based on an input dataset, a convolutional neural network is trained.
Therefore, Keras is used due to its simple network architecture definition methods.
The structure is almost symmetric, contains several dropouts and is purely convolutional.