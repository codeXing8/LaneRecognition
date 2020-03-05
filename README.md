## Algorithms for automotive lane markings recognition
### Deployment and sample data
#### Environment
It is recommended to use a virtual Python environment - created with e. g. Conda.
```bash
mkdir frames \
&& chmod +x install_dependencies.sh \
&& ./install_dependencies.sh
```

### List of methods
#### 1D-Line fitting to edge detection results under triangular mask
<p align="center">
    <img src="LaneRecognition001.gif" alt="Exemplary lane recognition algorithm results"/>
</p>
A sample video is loaded as an image stream and each image is sequentially analyzed.
Inside one frame (image), lanes are detected via edges.
By discarding short edges and averaging all kept edge positions, a 1D-polynomial fit (line) is used to determine slope and position of a single, averaged edge.
Based on both line parameters, the edge's length is increased from the lower horizontal image border to the image's vanishing point.

To exclude long but irrelevant edges surrounding the actual lane, the area between the lower image boundary and the vanishing points is masked.
A triangular mask excludes everything outside this region of interest (ROI).
Inside the ROI, the approach is applied as described above.    
Furthermore, general filter techniques like gradient limitation and sliding average considering the line parameters where used.

Since this approach uses a one-dimensional fit (linear), it is mostly useful for linear lane marking shapes.
In sharp curves, a polynomial fit of higher order or rather a clothoid fit would be necessary.
Since both shapes (especially the latter) are very common,
it will be kept in mind for upcoming methods presented in this repository.

#### Warped image edge detection and windowed 2D-polynomial fit