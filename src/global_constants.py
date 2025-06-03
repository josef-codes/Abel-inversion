import numpy as np

base_dir = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA'
#dir_1064_H0_1 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\1064 nm\H0_3_28_25\(1) 50-1000ns'
#dir_1064_H0_2 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\1064 nm\H0_3_28_25\(2) 1000 - 2000 ns'
dir_1064_H0_3 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\1064 nm\H0_3_28_25\(3) 2-5us'
#dir_1064_x4_H0 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\1064 nm\H0_3_28_25\reference x4'
#dir_1064_x2_H0 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\1064 nm\H0_3_28_25\reference x2'
#
dir_1064_H6_1 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\1064 nm\H6_3_31_25\(1) 50-1000ns'
dir_1064_H6_2 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\1064 nm\H6_3_31_25\(2) 1000-2000ns'
dir_1064_H6_3 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\1064 nm\H6_3_31_25\(3) 2-5us'
dir_1064_x4_H6 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\1064 nm\H6_3_31_25\reference x4'
dir_1064_x2_H6 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\1064 nm\H6_3_31_25\reference x2'
#
dir_2090_H6_1 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\H6\(1) 50-1000ns'
dir_2090_H6_2 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\H6\(2) 1000-2000ns'
dir_2090_H6_3 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\H6\(3) 2-5us'
#
#dir_2090_H0_1 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\H0\(1) 50-1000ns'
#dir_2090_H0_2 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\H0\(2) 1000-2000ns'
#dir_2090_H0_3 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\H0\(3) 2-5us'
#
#dir_2090_Cu_1 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\Cu\(1) 50-1000ns'
#dir_2090_Cu_2 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\Cu\(2) 1000-2000ns'
#dir_2090_Cu_3 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\Cu\(3) 2-5us'
#
dir_2090_x4_50ns = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\ref x4 Cu,H0-H2,H6 - 50-1000ns'
#dir_2090_x2_H0 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\H0\reference x2'
dir_2090_x2_H6 = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\H6\reference x2'
dir_2090_x4_H6_2us = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\ref x4 H6, 1-2us'
#dir_2090_x4_Cu_2us = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\ref x4 Cu, 1-2us'
#dir_2090_x2_Cu = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\Cu\reference x2'
t_µs_all = np.array([
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
    1.10, 1.30, 1.50, 1.70, 1.90, 2.10, 2.20, 2.60, 3.00, 3.40,
    3.80, 4.20, 4.60, 5.00, 6.00, 7.00, 8.00, 9.00, 10.0, 11.0,
    12.0, 13.0, 14.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0,
    85.0, 95.0,
    100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0,
    1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0,
    1800.0, 1900.0, 2000.0
], dtype=np.float64)

t_µs_el_dens = np.array([
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
    1.10, 1.30, 1.50, 1.70, 1.90, 2.10,
    2.20, 2.60, 3.00, 3.40, 3.80, 4.20, 4.60, 5.00,
    6.00, 7.00, 8.00, 9.00, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0
], dtype=np.float64)

t_us_H0_1064ns = np.array([
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
    1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.0, 2.10,
    2.0, 2.20, 2.40, 2.60, 2.80, 3.00, 3.20, 3.40, 3.60, 3.80, 4.00, 4.20, 4.40, 4.60, 4.80, 5.00,
    5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.0, 10.50, 11.0, 11.50, 12.0, 12.50, 13.0, 13.50, 14.0, 14.50, 15.0
], dtype=np.float64)