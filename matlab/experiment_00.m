clear;
clc;

% Change these file paths to match your path setup
dataFolder_fusedng = fullfile('/home','server', 'SAR', 'geo_dataset','Concord', 'moving');
dataFolder_reference = fullfile('/home','server', 'SAR', 'geo_dataset','Concord', 'fixed');
dataFolder_fused_output = fullfile('/home','arwillis', 'geo_dataset_exp');

% dataFolder_moving = fullfile('..','src','gpu', 'testImages','reg','sar');
% dataFolder_reference = fullfile('..','src','gpu', 'testImages','reg','gmap');
gt_data = readtable(fullfile('/home','server', 'SAR', 'geo_dataset','Concord', 'rosgeoregistration_2022_12_21_19_19_13.log'));
execute_binary='../cmake-build-debug/src/cpu/multispectral_ImageMatcher';

imds_reference = imageDatastore(dataFolder_reference, 'IncludeSubfolders',true,'LabelSource','foldernames');
imds_moving = imageDatastore(dataFolder_fusedng, 'IncludeSubfolders',true,'LabelSource','foldernames');

arg_reference_image='--i_ref';
arg_moving_image='--i_mov';
arg_fused_output_image='-f';
image_ref='';
image_mov='';

estH_list = [];
new_gtH = [];
scale_list = [];
runtime_list = [];

for fileIdx=1:length(imds_moving.Files)
    time_start = tic;
    image_ref=imds_reference.Files(fileIdx);
    image_mov=imds_moving.Files(fileIdx);
    indices=find('/' == image_ref{1});
    image_out_fused = strcat(dataFolder_fused_output,'/',image_ref{1}(indices(end)+1:end-9),'fused.png');
    command=strcat(execute_binary," ", ...
        arg_reference_image," ",image_ref," ", ...
        arg_moving_image," ",image_mov, " ", ...
        arg_fused_output_image," ",image_out_fused)
    
    [status,cmdout] = system(command);
    % add output of homography to cuGridSearch
    runtime = toc(time_start);
    result = regexp(cmdout,"Answer\[([^\]]*)]\s+Scale_factor\[([^\]]*)]", "tokens");
    estH = reshape(str2num(result{1}{1}),1,[]);
    estH = [estH 1]; % Only has 8 values originally, needs h33 to be 1
    scale_factor = reshape(str2num(result{1}{2}),1,[]);
    estH_list = [estH_list; estH];
    scale_list = [scale_list; scale_factor];
    runtime_list = [runtime_list; runtime];
    
    Hvals = table2array(gt_data(fileIdx,18:26));
    gt_H = reshape(Hvals, 3,3)';
    scale = scale_list(fileIdx, :);
    S = [scale(1) 0 0; 0 scale(2) 0; 0 0 1];
    % Formula: B' = (S * H * S^-1) * S * A
    resized_gtH = S * gt_H;
    temp_gtH = reshape(resized_gtH', 1,[]);
    new_gtH = [new_gtH; temp_gtH];
    est_H1 = reshape(estH, 3,3)';
    
    aaa = 1;
end



new_gtH = [];
for fileIdx=1:length(imds_moving.Files)
    gt_H = reshape(gt_data(fileIdx, 2:10), 3,3)';
    scale = scale_list(fileIdx, :);
    S = [scale(1) 0 0; 0 scale(2) 0; 0 0 1];
    % Formula: B' = (S * H * S^-1) * S * A
    resized_gtH = S * gt_H;
    temp_gtH = reshape(resized_gtH', 1,[]);
    new_gtH = [new_gtH; temp_gtH];
end