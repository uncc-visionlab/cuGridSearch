clear;
clc;

% DATASET_INDEX=1;
% SET NUM_FILES = 100000 TO DO ALL FILES
NUM_FILES = 100000;
%NUM_FILES = 2;
USER_FOLDER = 'arwillis';
ERROR_THRESHOLD = 0.18;

datasetName{1}='EPIC';
datafileGTName{1}='rosgeoregistration_2022_12_21_23_38_22.log';
datasetName{2}='Concord';
datafileGTName{2}='rosgeoregistration_2022_12_21_19_19_13.log';
datasetName{3}='Isleta';
datafileGTName{3}='rosgeoregistration_2022_12_21_21_28_42.log';

for DATASET_INDEX=1:3
    % Change these file paths to match your path setup
    % Willis - UNIVERSITY SITE CONFIG
    %dataFolder_root = fullfile('/home', USER_FOLDER, 'CLionProjects', 'georeg_exps');
    %dataFolder_root = fullfile('/home','server', 'SAR');
    % Willis - HOME SITE CONFIG
    dataFolder_root = fullfile('/home', USER_FOLDER, 'CLionProjects', 'georeg_exps');
    dataFolder_results = fullfile('/home', USER_FOLDER,  'CLionProjects', 'georeg_exps', 'results');
    
    dataFolder_dataset = fullfile('geo_dataset', datasetName{DATASET_INDEX});
    dataFolder_moving = fullfile(dataFolder_root, dataFolder_dataset, 'moving');
    dataFolder_reference = fullfile(dataFolder_root, dataFolder_dataset, 'fixed');
    dataFolder_fused_output = fullfile(dataFolder_results, datasetName{DATASET_INDEX});
    dataFolder_compared_fused_output = fullfile(dataFolder_results, datasetName{DATASET_INDEX},'comparisons');
    
    gt_data = readtable(fullfile(dataFolder_root, dataFolder_dataset, datafileGTName{DATASET_INDEX}));
    
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
    corner_error_statistics = [];
    
    f1_handle = figure('visible','off');
    
    skip = round(length(imds_moving.Files)/(NUM_FILES));
    if (skip == 0)
        skip = 1;
    end
    
    for fileIdx=1:skip:length(imds_moving.Files)
        time_start = tic;
        image_ref=imds_reference.Files(fileIdx);
        image_mov=imds_moving.Files(fileIdx);
        indices=find('/' == image_ref{1});
        image_out_fused = strcat(dataFolder_fused_output,'/',image_ref{1}(indices(end)+1:end-9),'fused.png');
        %command=strcat(execute_binary," ", ...
        %    arg_reference_image," ",image_ref," ", ...
        %    arg_moving_image," ",image_mov, " ", ...
        %    arg_fused_output_image," ",image_out_fused)
        command=strcat(execute_binary," ", ...
            arg_reference_image," ",image_ref," ", ...
            arg_moving_image," ",image_mov);
        
        [status,cmdout] = system(command);
        % add output of homography to cuGridSearch
        runtime = toc(time_start);
        grid_values_str = regexp(cmdout,"\{(.*?)\}", "tokens");
        grid_values.start = str2num(grid_values_str{1}{1});
        grid_values.stop = str2num(grid_values_str{2}{1});
        grid_values.stepsize = str2num(grid_values_str{3}{1});
        grid_values.numsteps = str2num(grid_values_str{4}{1});
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
        
        scale = scale_list(end, :);
        S = [scale(1) 0 0; 0 scale(2) 0; 0 0 1];
        
        % Formula: B' = (S * H * S^-1) * S * A
        H_ground_truth_rescaled = inv(S) * H_ground_truth * S;
        H_ground_truth_rowvec = reshape(H_ground_truth_rescaled', 1,[]);
        H_ground_truth_list = [H_ground_truth_list; H_ground_truth_rowvec];
        
        [rows, cols, ~] = size(I_fixed_scaled);
        
%         tform = maketform('projective', [scale(1) 0 0; 0 scale(2) 0; 0 0 1]');
%         I_fixed_scaled2 = imtransform(I_fixed, tform, 'XData',[1 cols],'YData',[1 rows], 'XYScale',1);
        
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
        
        [rows, cols, ~] = size(I_fixed_scaled);
        I_fused_scaled = zeros(rows, cols, 3, 'uint8');
        I_fused_scaled(:,:,2) = uint8(I_fixed_scaled(:,:,1));
        I_fused_scaled(:,:,3) = uint8(2*I_moving_scaled_estimated(:,:,1));
        
        basename = image_ref{1}(indices(end)+1:end-9);
        image_out_fused_ground_truth = fullfile(dataFolder_fused_output,strcat(basename,'fused_gt.png'));
        image_out_fused_estimated = fullfile(dataFolder_fused_output,strcat(basename,'fused_est.png'));
        imwrite(I_fused, image_out_fused_ground_truth);
        imwrite(I_fused_scaled, image_out_fused_estimated);
        
        set(0,'CurrentFigure',f1_handle);
        subplot(1,2,1), imshow(I_fused), title('Ground Truth');
        subplot(1,2,2), imshow(I_fused_scaled), title('Estimated');
        image_out_comp_fused = fullfile(dataFolder_compared_fused_output,strcat(basename,'fused_comp.png'));
        saveas(f1_handle, image_out_comp_fused)
        
        [rows_mov, cols_mov, ~] = size(I_moving);
        % coordinates of box listed in clockwise order
        corners_pre_homography = [0, 0, 1; 0, cols_mov, 1; rows_mov, cols_mov, 1; rows_mov, 0, 1];

        [rows_fix, cols_fix, ~] = size(I_fused);
        corners_normalized_ground_truth = corners_pre_homography*H_ground_truth';
        corners_normalized_ground_truth = corners_normalized_ground_truth./corners_normalized_ground_truth(:,3);
        corners_normalized_ground_truth = corners_normalized_ground_truth./(ones(4,1)*[cols_fix, rows_fix, 1]);

        [rows_fix, cols_fix, ~] = size(I_fused_scaled);
        corners_normalized_estimated = corners_pre_homography*H_estimated';
        corners_normalized_estimated = corners_normalized_estimated./corners_normalized_estimated(:,3);
        corners_normalized_estimated = corners_normalized_estimated./(ones(4,1)*[cols_fix, rows_fix, 1]);
        
        corners_normalized_error = corners_normalized_ground_truth - corners_normalized_estimated;
        avg_corner_normalized_error_mag = median(sqrt(sum(corners_normalized_error.^2, 2)));
        corner_error_statistics = [corner_error_statistics; avg_corner_normalized_error_mag];
        if (avg_corner_normalized_error_mag < ERROR_THRESHOLD)
            result_str = 'ACCEPTABLE';
        else
            result_str = 'NOT ACCEPTABLE';
        end
        output_str = sprintf('Match %s dataset, image index %d, average normalized corner error = %f, result = %s.\n', ...
            datasetName{DATASET_INDEX}, fileIdx-1, avg_corner_normalized_error_mag, result_str);
        fprintf(1, output_str);
        
        % record experimental data to archive
        
        dataset(DATASET_INDEX).match(fileIdx).filename_moving = image_mov;
        dataset(DATASET_INDEX).match(fileIdx).filename_fixed = image_ref;
        dataset(DATASET_INDEX).match(fileIdx).grid = grid_values;
        dataset(DATASET_INDEX).match(fileIdx).scale_factor = scale_factor;
        dataset(DATASET_INDEX).match(fileIdx).image_size_moving = size(I_moving);
        dataset(DATASET_INDEX).match(fileIdx).image_size_fixed = size(I_fixed);
        dataset(DATASET_INDEX).match(fileIdx).image_size_fixed_scaled = size(I_fixed_scaled);
        dataset(DATASET_INDEX).match(fileIdx).H_ground_truth = H_ground_truth;
        dataset(DATASET_INDEX).match(fileIdx).H_estimated = H_estimated;
        dataset(DATASET_INDEX).match(fileIdx).corners_normalized_ground_truth = corners_normalized_ground_truth;
        dataset(DATASET_INDEX).match(fileIdx).corners_normalized_estimated = corners_normalized_estimated;
        dataset(DATASET_INDEX).match(fileIdx).avg_corner_normalized_error_mag = avg_corner_normalized_error_mag;
        dataset(DATASET_INDEX).match(fileIdx).runtime = runtime;
    end
end
save('exp_results.mat','dataset')

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