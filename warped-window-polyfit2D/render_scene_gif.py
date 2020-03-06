# Import dependencies
from moviepy.editor import ImageSequenceClip

# Render scene as .gif file function
def render_scene_gif(filenames, fps):
    # Render .gif file
    clip = ImageSequenceClip(filenames, fps=fps)
    clip.write_gif('LaneRecognition002.gif', fps=fps)
    return 0