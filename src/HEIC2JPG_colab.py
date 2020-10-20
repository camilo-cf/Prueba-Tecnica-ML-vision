"""
Author: Camilo A. Cáceres - 2020 
MIT LICENCE

Support file for transform HEIC files to JPG
- Implemented in Google Colab due to its imposibility of use in Windows
"""

# Google colab install
! pip install pyheif

from PIL import Image
import pyheif
import glob

# Google colab create a directory
! mkdir JPEG

def heic2jpg(filename):
    """
    Function to convert a HEIC file to JPG

    Args:
    filename (str): name of the file to convert

    """

    if (".HEIC" in filename) or (".png" in filename):
        heif_file = pyheif.read(filename)
        image = Image.frombytes(
                                heif_file.mode, 
                                heif_file.size, 
                                heif_file.data,
                                "raw",
                                heif_file.mode,
                                heif_file.stride,
                                )
        
        image.save("JPEG/"+str(filename.split(".")[0])+".jpg", "JPEG")

# Read the uploaded files in the colab 
files = glob.glob('/content/*.*')

# Get the name of the files in the right format
file_names = [x.split("/")[-1] for x in files]

# Covert all the files to the desired format
for each_file in file_names:
    heic2jpg(each_file)

# Create the compressed resulting files (compress JPEG directory) for download the results easily
! tar -zcvf pictures.tar.gz JPEG