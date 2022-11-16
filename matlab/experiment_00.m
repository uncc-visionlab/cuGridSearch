clear;
clc;

dataFolder_moving = fullfile('..','data','dataset_SAR_EO','moving');
imds_moving = imageDatastore(dataFolder_moving, 'IncludeSubfolders',true,'LabelSource','foldernames');
dataFolder_reference = fullfile('..','data','dataset_SAR_EO','fixed');
imds_reference = imageDatastore(dataFolder_reference, 'IncludeSubfolders',true,'LabelSource','foldernames');

gt_data = csvread('../data/dataset_SAR_EO/gt_dataset_homography.csv');
execute_binary='../cmake-build-debug/src/cpu/multispectral_ImageMatcher';
arg_reference_image='--i_ref';
arg_moving_image='--i_mov';
image_ref='';
image_mov='';
for fileIdx=1:length(imds_moving.Files)
    time_start = tic;
    image_ref=imds_reference.Files(fileIdx);
    image_mov=imds_moving.Files(fileIdx);
    command=strcat(execute_binary," ", ...
        arg_reference_image," ",image_ref," ", ...
        arg_moving_image," ",image_mov)
    
    [status,cmdout] = system(command);
    % add output of homography to cuGridSearch
    result = regex(cmdout,'.*Answer[$1].*')
    runtime = toc(tStart);
end

