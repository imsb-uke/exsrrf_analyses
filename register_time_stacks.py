# Import dependencies
import numpy as np
from nptyping import NDArray
import cv2
from skimage.metrics import structural_similarity, mean_squared_error
from skimage import img_as_ubyte
from tifffile import imread
from skimage.filters import gaussian
from image_registration import chi2_shift
from scipy.ndimage import shift
from tifffile import imsave
from glob import glob
from skimage.exposure import match_histograms
import os
import pandas as pd

src_path = '/path_to_data/'

src_files = list = [fn for fn in glob(os.path.join(src_path, '*.tif'))]

out_path = '/output_directory/'

reference_channel: int = 1
global_reference_path = None

# Optionally restrict the number of images that will be transformed
n_files_max: int = len(src_files)

threshold_percentile: int = 90
registration_method: str = 'correlation'

# Lists to save values for overall dataframe
file_names = []

mean_mssim_before = []
mean_mssim_after = []
last_mssim_before = []
last_mssim_after = []

mean_mse_before = []
mean_mse_after = []
last_mse_before = []
last_mse_after = []

mean_shift = []
last_shift = []

# Iterate over different image stacks
for src_file_path in src_files[:n_files_max]:

    img_identifier = str(src_file_path.split('/')[-1].split('.tif')[0])
    file_names.append(img_identifier)

    print('Converting image ' + img_identifier)
    log_file = open(out_path + img_identifier + '_log.txt', 'w')

    try:
        img_src = imread(src_file_path)
    except:
        print('Problem processing file ', img_identifier)
        continue

    print(img_src.shape)
    log_file.write(str(img_src.shape))

    if len(img_src.shape) == 2:
        # If no time and channel axes, add them
        img_src = np.reshape(img_src, (1, img_src.shape[0], img_src.shape[1], 1))
    elif len(img_src.shape) == 3:
        # Time but not channels: add channel
        img_src = np.reshape(img_src, (img_src.shape[0], img_src.shape[1], img_src.shape[2], 1))
    elif len(img_src.shape) == 4:
        # Channels and time are present, flip axes
        img_src = np.moveaxis(img_src, 1, -1)

    global_reference = False
    global_reference_u8 = None
    global_reference_float = None
    global_reference_mssim = None

    mssim_before = []
    mssim_after = []
    mse_before = []
    mse_after = []

    x_shifts = []
    y_shifts = []

    timepoints = []
    image_timepoints = []
    image_timepoints_smoothed = []

    print(img_src.shape)
    log_file.write(str(img_src.shape))

    for timepoint in range(0, img_src.shape[0]):

        print('Timepoint ' + str(timepoint) + '.')
        log_file.write('Timepoint ' + str(timepoint) + '.' + '\n')

        channel_images = []

        # The first image of the stack becomes the reference image
        if global_reference is False:

            img_dest = img_src[timepoint, :, :, reference_channel]

            global_reference_mssim = img_dest

            global_reference = gaussian(img_dest, sigma=1)
            global_reference[global_reference < np.percentile(global_reference, threshold_percentile)] = 0

            # Save before absolute thresholding that depends on the histogram
            global_reference_float = global_reference

            global_reference_u8 = img_as_ubyte(global_reference)

            if len(global_reference_u8.shape) == 2:
                # If no time and channel axes, add them
                global_reference_time_channels = np.reshape(global_reference_u8,
                                                            (1, global_reference_u8.shape[0],
                                                             global_reference_u8.shape[1], 1))

            if global_reference_path is None:

                for channel in range(img_src.shape[3]):
                    channel_images.append(img_src[timepoint, :, :, channel])
                image_timepoints_smoothed.append(global_reference_u8)

        # Register to the reference image
        elif global_reference_u8 is not None:

            local_reference = gaussian(img_src[timepoint, :, :, reference_channel], sigma=1)

            # Match global reference shape to calculate mssim later
            local_reference = cv2.resize(local_reference, (global_reference.shape[1], global_reference.shape[0]),
                                         interpolation=cv2.INTER_AREA)

            # Threshold local reference channel
            local_reference[local_reference < np.percentile(local_reference, threshold_percentile)] = 0

            # Match the histogram to the also previously thresholded global reference
            local_reference = match_histograms(local_reference, global_reference_float)

            x_off, y_off = chi2_shift(global_reference, local_reference, return_error=False)
            print(y_off, x_off)
            log_file.write('y shift: ' + str(y_off) + ', x shift: ' + str(x_off) + '\n')

            y_shifts.append(y_off)
            x_shifts.append(x_off)

            for channel in range(img_src.shape[3]):

                channel_image = img_src[timepoint, :, :, channel]

                channel_image = cv2.resize(channel_image, (global_reference.shape[1], global_reference.shape[0]),
                                           interpolation=cv2.INTER_AREA)

                if channel == reference_channel:
                    limits_x1 = max(0, -int(x_off))
                    limits_x2 = min(global_reference.shape[1], int(global_reference.shape[1] - x_off))

                    limits_y1 = max(0, -int(y_off))
                    limits_y2 = min(global_reference.shape[0], int(global_reference.shape[0] - y_off))

                    # Save structural similarity and MSE before registration
                    mssim_before.append(
                        structural_similarity(global_reference_mssim[limits_y1:limits_y2, limits_x1:limits_x2],
                                              channel_image[limits_y1:limits_y2, limits_x1:limits_x2]))
                    mse_before.append(
                        mean_squared_error(global_reference_mssim[limits_y1:limits_y2, limits_x1:limits_x2],
                                           channel_image[limits_y1:limits_y2, limits_x1:limits_x2]))

                channel_image: NDArray = shift(channel_image, shift=(-y_off, -x_off), mode='constant')

                if channel == reference_channel:
                    # Save structural similarity and MSE after registration
                    mssim_after.append(
                        structural_similarity(global_reference_mssim[limits_y1:limits_y2, limits_x1:limits_x2],
                                              channel_image[limits_y1:limits_y2, limits_x1:limits_x2]))
                    mse_after.append(
                        mean_squared_error(global_reference_mssim[limits_y1:limits_y2, limits_x1:limits_x2],
                                           channel_image[limits_y1:limits_y2, limits_x1:limits_x2]))

                channel_images.append(channel_image)

        if len(channel_images):
            image_timepoints.append(channel_images)

        timepoints.append(timepoint)

    mean_mssim_before.append(np.mean(np.array(mssim_before)))
    mean_mssim_after.append(np.mean(np.array(mssim_after)))
    last_mssim_before.append(np.array(mssim_before)[-1])
    last_mssim_after.append(np.array(mssim_after)[-1])

    mean_mse_before.append(np.mean(np.array(mse_before)))
    mean_mse_after.append(np.mean(np.array(mse_after)))
    last_mse_before.append(np.array(mse_before)[-1])
    last_mse_after.append(np.array(mse_after)[-1])

    shifts = np.linalg.norm([np.array(abs(np.array(x_shifts))), np.array(abs(np.array(y_shifts)))], axis=0)

    mean_shift.append(np.mean(shifts))
    last_shift.append(shifts[-1])

    img_tpts = np.array(image_timepoints).swapaxes(0, 1)

    # Save the image timestack as a tiff file for each channel
    for index, channel in enumerate(img_tpts):
        print(channel.shape)
        imsave(out_path + img_identifier + '_registered_ch' + str(index) + '.tif',
               channel,
               metadata={'axes': 'TXY'}, dtype=np.uint16)

    log_file.close()

    break

mean_mssim_diff = np.array(mean_mssim_before) - np.array(mean_mssim_after)
last_mssim_diff = np.array(last_mssim_before) - np.array(last_mssim_after)

mean_mse_diff = np.array(mean_mse_before) - np.array(mean_mse_after)
last_mse_diff = np.array(last_mse_before) - np.array(last_mse_after)

# Save evaluation data of registrations
overall_data = pd.DataFrame({'File': file_names,  # 'Batch': batch,
                             'Mean_shift': mean_shift, 'Last_shift': last_shift,
                             'Mean_MSSIM_diff': mean_mssim_diff, 'Last_MSSIM_diff': last_mssim_diff,
                             'Mean_MSE_diff': mean_mse_diff, 'Last_MSE_diff': last_mse_diff})
overall_data.to_csv(out_path + 'Overall_MSSIM_MSE_shifts.csv', index=False)
