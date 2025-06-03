# Image processing script for plasma Electron density reconstruction from plasma interferograms
The script uses the diffraction peak filtration technique to retrieve the phase image, symmetrizes the image, then uses one of PyAbel's Abel inversion algorithms to retrieve the change in index of refraction map of the plasma, then, from Drude's model equation, calculates the electron density map.

## Installation
When installing PyAbel package, if an error occurs, one must replace the nested function with:
- return input.decode("utf-8", errors='ignore')
Then return the function to its previous state.

- Code has been tested on Windows 10, IDE PyCharm

## Source
Details are explained in the author's Diploma Thesis: *HRDLIČKA, Josef. Detection and characterization of laser-induced plasmas. Master's Thesis. Jakub BUDAY (supervisor). Brno: Brno University of Technology, Faculty of Mechanical Engineering, 2025.* [Link to thesis](https://www.vut.cz/en/students/final-thesis/detail/166167).
