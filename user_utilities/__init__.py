import imageio
import os
import re


def make_gif(folder_path: str, output_file: str, fps=30, compress=True):
    """Saves a gif of all the files in a given folder

    Args:
        folder_path: The folder with all the input images with names that start or end with their
            order in the list. For example, Picture1.png, Picture2.png, ..., Picture129.png.
            Folder names should end in .png or .jpg
        output_file: The name of the gif. It will be saved in the current working directory if
            it is not an absolute path.
        fps: Frames per second of the animation.
        compress: Whether or not to compress the result

    """
    duration = 1/fps
    file_names = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                  if os.path.isfile(os.path.join(folder_path, f))
                  and (f.endswith(".png") or f.endswith(".jpg"))]
    file_names.sort(key=lambda var: [
        int(x) if x.isdigit() else x
        for x in re.findall(r'[^0-9]|[0-9]+', var)
    ])

    with imageio.get_writer(output_file, mode='I', fps=fps) as writer:
        for filename in file_names:
            image = imageio.imread(filename)
            writer.append_data(image)

    if compress:
        pass #optimize(output_file)





