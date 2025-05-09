"""
Main code to run the Abel inversion.
"""

# ---- IMPORTS from my functions ----
# add path to make sure the imports happen:
import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..')) # Go up two levels from current path (from src/ to project root)
sys.path.append(project_root)
# or: sys.append(r"C:\Users\User\z\Desktop\WUT\Diplomka\ZPRACOVÁNÍ\Data testing\processing_project")
import global_utils
import constants # print(constants.__file__)
import global_functions_im
import utils
import functions_image_crop
import functions_phase_shift
import functions_abel
import functions_electron_density_analysis

# ---- other IMPORTS ----
