clear;
clc;

% Change these file paths to match your path setup
dataFolder_root = fullfile('/home','arwillis', 'CLionProjects', 'georeg_exps');
dataFolder_dataset = fullfile('geo_dataset', 'Concord');
dataFolder_moving = fullfile(dataFolder_root, dataFolder_dataset, 'moving');
dataFolder_reference = fullfile(dataFolder_root, dataFolder_dataset, 'fixed');
dataFolder_fused_output = fullfile(dataFolder_root, 'results', 'Concord');
gt_data = readtable(fullfile(dataFolder_root, dataFolder_dataset, 'rosgeoregistration_2022_12_21_19_19_13.log'));

% dataFolder_moving = fullfile('..','src','gpu', 'testImages','reg','sar');
% dataFolder_reference = fullfile('..','src','gpu', 'testImages','reg','gmap');
execute_binary='../cmake-build-debug/src/cpu/multispectral_ImageMatcher';

imds_reference = imageDatastore(dataFolder_reference, 'IncludeSubfolders',true,'LabelSource','foldernames');
imds_moving = imageDatastore(dataFolder_moving, 'IncludeSubfolders',true,'LabelSource','foldernames');

arg_reference_image='--i_ref';
arg_moving_image='--i_mov';
arg_fused_output_image='-f';
image_ref='';
image_mov='';

H_estimated_list = [];
H_ground_truth_list = [];
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
    H_estimated = reshape(str2num(result{1}{1}),1,[]);
    H_estimated = [H_estimated 1]; % Only has 8 values originally, needs h33 to be 1
    scale_factor = reshape(str2num(result{1}{2}),1,[]);
    H_estimated_list = [H_estimated_list; H_estimated];
    H_estimated = reshape(H_estimated, 3, 3)';
    scale_list = [scale_list; scale_factor];    
    runtime_list = [runtime_list; runtime];
    
    I_fixed = imread(image_ref{1});
    I_moving = imread(image_mov{1});
    I_fixed_scaled = imread('image_fixed_resized.png');
    
    Hvals = table2array(gt_data(fileIdx,18:26));
    H_ground_truth = reshape(Hvals, 3,3)';

    scale = scale_list(fileIdx, :);
    S = [scale(1) 0 0; 0 scale(2) 0; 0 0 1];

    % Formula: B' = (S * H * S^-1) * S * A
    H_ground_truth_rescaled = inv(S) * H_ground_truth * S 
    H_ground_truth_rowvec = reshape(H_ground_truth_rescaled', 1,[]);
    H_ground_truth_list = [H_ground_truth_list; H_ground_truth_rowvec];
    
    [rows, cols, ~] = size(I_fixed_scaled);

    tform = maketform('projective', [scale(1) 0 0; 0 scale(2) 0; 0 0 1]');
    I_fixed_scaled2 = imtransform(I_fixed, tform, 'XData',[1 cols],'YData',[1 rows], 'XYScale',1);

    tform = maketform('projective', H_estimated');
    I_moving_scaled_estimated = imtransform(I_moving, tform, 'XData',[1 cols],'YData',[1 rows], 'XYScale',1);
    
    tform = maketform('projective', H_ground_truth_rescaled');
    I_moving_ground_truth_scaled = imtransform(I_moving, tform, 'XData',[1 cols],'YData',[1 rows], 'XYScale',1);

    [rows, cols, ~] = size(I_fixed);
    tform = maketform('projective', H_ground_truth');
    I_moving_ground_truth = imtransform(I_moving, tform, 'XData',[1 cols],'YData',[1 rows], 'XYScale',1);
    
    I_fused = zeros(rows, cols, 3, 'uint8');
    I_fused(:,:,2) = uint8(I_fixed(:,:,1));
    I_fused(:,:,3) = uint8(2*I_moving_ground_truth(:,:,1));
    subplot(1,2,1), imshow(I_fused);
    
    [rows, cols, ~] = size(I_fixed_scaled);
    I_fused_scaled = zeros(rows, cols, 3, 'uint8');
    I_fused_scaled(:,:,2) = uint8(I_fixed_scaled(:,:,1));
    I_fused_scaled(:,:,3) = uint8(2*I_moving_scaled_estimated(:,:,1));
    subplot(1,2,2), imshow(I_fused_scaled);
    
    aaa=1;
end



H_ground_truth_list = [];
for fileIdx=1:length(imds_moving.Files)
    H_ground_truth = reshape(gt_data(fileIdx, 2:10), 3,3)';
    scale = scale_list(fileIdx, :);
    S = [scale(1) 0 0; 0 scale(2) 0; 0 0 1];
    % Formula: B' = (S * H * S^-1) * S * A
    H_ground_truth_rescaled = S * H_ground_truth;
    H_ground_truth_rowvec = reshape(H_ground_truth_rescaled', 1,[]);
    H_ground_truth_list = [H_ground_truth_list; H_ground_truth_rowvec];
end


function transformImage(I_src, I_dest, H)
    [rows_dest, cols_dest] = size(I_dest);
    [rows_src, cols_src] = size(I_src);
    [x_src, y_src] = meshgrid(1:1:cols_src,1:1:rows_src);
    %[x_dest, y_dest] = meshgrid(1:1:cols_dest,1:1:rows_dest);
    %I_src = I_src(:);
    homogeneous_coords = [x_dest(:), y_dest(:) ones(rows_dest*cols_dest,1)];
    homogeneous_coords_xformed = homogeneous_coords*inv(H)';
    homogeneous_coords_xformed = homogeneous_coords_xformed./(homogeneous_coords_xformed(:,3)*ones(1,3));
    homogeneous_coords(homogeneous_coords_xformed(:,1) < 1 | ...
        homogeneous_coords_xformed(:,1) > cols_src | ...
        homogeneous_coords_xformed(:,2) < 1 | ...
        homogeneous_coords_xformed(:,2) > rows_src) =[];
    homogeneous_coords_xformed(homogeneous_coords_xformed(:,1) < 1 | ...
            homogeneous_coords_xformed(:,1) > cols_src | ...
            homogeneous_coords_xformed(:,2) < 1 | ...
            homogeneous_coords_xformed(:,2) > rows_src) =[];
    
    I_xformed_vals = interp2(x_src, y_src, I_src,  ...
        homogeneous_coords_xformed(:,1), homogeneous_coords_xformed(:,2));
    I_new = reshape(I_xformed_vals,[rows, cols]);
end