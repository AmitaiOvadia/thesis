function label_joints_3d_V2(boxPath, skeletonPath)
addpath("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\main datasets\head_tail_dataset")
addpath("G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled dataset")
%LABEL_JOINTS GUI to click on images to yield a graph.
% Usage:
%   label_joints(boxPath)
%   label_joints(boxPath, skeletonPath)
%
% See also: make_template_skeleton
% predictions_path = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\sigma_2_14_points\predictions_for_labeling.h5";
% per_wing_7_points = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\per_wing\7_points_together\per_wing_model_filters_64_sigma_2_trained_by_141_frames\predictions_over_movie.h5";
predictions_xx = "C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\models\find 8 pts per wing\per_wing_16_pts_28_1\predict_over_video.h5";
predictions_16_pts = "C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\movie_test_set_new\predictions_on_movie.mat";
predictions_path = predictions_16_pts;
%% Startup
% addpath(genpath('deps'))

% mfl_rep_path='C:\git_reps\micro_flight_lab';
% addpath(fullfile(mfl_rep_path,'Insect analysis'))
% addpath(fullfile(mfl_rep_path,'Utilities'))
% get camera calibration
% easyWandData=load( uibrowse('*.mat',[],'Select easywand file'));

im_height=800;
easywand_path = uigetfile('*.mat',[],'Select calibration file');
easyWandData=load(easywand_path);

frame_size=[];
allCams=HullReconstruction.Classes.all_cameras_class(easyWandData.easyWandData);
n_cams=length(allCams.cams_array);
fundMats=cat(3,allCams.Fundamental_matrices,permute(allCams.Fundamental_matrices,[2 1 3]));
coupless=[nchoosek(1:n_cams,2);fliplr(nchoosek(1:n_cams,2))];

% Ask for path to data file
if nargin < 1 || isempty(boxPath); boxPath = uibrowse('*.h5',[],'Select box HDF5 file'); end

% Params
if nargin < 2; skeletonPath = []; end
recreate_labels = nargin > 1; % force recreate labels file

% Settings (saved in *.labels.mat file)
global config;
config = struct();
config.dsetName = '/box';
config.nodeSize = 10; % size of draggable markers
config.defaultNodeColor = [1 0 0]; % default color of movable nodes
config.initializedNodeColor = [1 1 0]; % color of initialized nodes
config.labeledNodeColor = [0 1 0]; % color of movable nodes with user input
config.initialFrame = 1; % first frame displayed
config.shuffleFrames = false; % shuffle frame order
config.autoSave = true; % save before going to a new frame
config.clickNearest = false; % true = click moves nearest node; false = selected node
config.draggable = true; % false = cannot drag joint markers
config.altArrowsToMoveNodes = true; % false = arrow keys move nodes, alt+arrows changes frames
config.zoomBoxFrames = [-250, 250]; % number of frames in the status zoomed in box (pre, post)
config.imgFigPos = [835 341 709 709]; % main labeling figure window
config.ctrlFigPos = [1545 342 374 708]; % control/reference window
config.statusFigPos = [836 33 1081 277]; % status bars and settings window

%%
% Initialize labeling session
box = [];
cropZone=[];
numNodes = [];
numFrames = [];
numLabeled = [];
numChannels=[];
global labels;

% Loads or creates *.labels.mat and populates config
initializeLabels();

% Pre-shuffle frames for shuffle mode
shuffleIdx = randperm(numFrames);

% Set status colormap colors
statusCmap = {
    config.defaultNodeColor
    config.initializedNodeColor
    config.labeledNodeColor
    };
for k = 1:numel(statusCmap)
    if ischar(statusCmap{k}); statusCmap{k} = colorCode2rgb(statusCmap{k}); end
end
statusCmap = cellcat(statusCmap,1);

% Zoom box convenience (compute window)
if isscalar(config.zoomBoxFrames); config.zoomBoxFrames = round([-0.5 0.5] .* config.zoomBoxFrames); end
zoomBoxWindow = config.zoomBoxFrames(1):config.zoomBoxFrames(2);

    function initializeLabels()
        labels = struct();

        % Metadata
        labels.boxPath = boxPath;
        labels.savePath = repext(boxPath, '.labels.mat');

        % Ask for path to skeleton file
        if isequal(skeletonPath,true) || ~exists(labels.savePath)
            skeletonPath = uibrowse('*.mat',[],'Select skeleton MAT file');
        end

        % Open box file
        box = double(h5file(boxPath, config.dsetName));
        if max(box, [], 'all') > 1
            box = box/255;
        end
        cropZone=h5file(boxPath, '/cropzone');
        numFrames = size(box,4);
        frame_size=size(box,1);
        numChannels = size(box,3);
      

        stic;
        if ~exists(labels.savePath) || recreate_labels
            % Load template skeleton
            labels.skeletonPath = skeletonPath;
            labels.skeleton = load(skeletonPath);

            % Initialize custom defaults container
            labels.initialization = NaN(numel(labels.skeleton.nodes), 2,n_cams, numFrames, 'single');

            % Try using initialization built into the HDF5 file
            try
                labels.initialization = h5read(boxPath, '/initialization');
                labels.initialization_metadata = h5att2struct(boxPath, '/initialization');

                printf('Using pre-initialized joint predictions.')
            catch
            end

            % Initialize user labels
            labels.positions = NaN(numel(labels.skeleton.nodes), 2, n_cams,numFrames, 'single');
            % (18,2,4,151)
            % ----------------------------------------------------------------------------------------
            % labeled_positions_1_7_9_15 = load("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\main_dataset_1000_frames_5_channels\trainset 18 points\labels_for_14_points.labels.mat"); 
            % labeled_positions_8_16_17_18 = load("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\models\5_channels_3_times_2_masks\main_dataset_1000_frames_5_channels\trainset 18 points\labels_for_wing_joints_and_head_tail.labels.mat");
            % labeled_positions_1_7_9_15 = labeled_positions_1_7_9_15.positions;
            % labeled_positions_8_16_17_18 = labeled_positions_8_16_17_18.positions;
            % labels.positions([1,2,3,4,5,6,7, 9,10,11,12,13,14,15],:,:,:) = labeled_positions_1_7_9_15;
            % labels.positions([18,17,8,16],:,:,:) = labeled_positions_8_16_17_18;
            labels.positions = load("G:\My Drive\Amitai\one halter experiments\roni dark 60ms\labeled dataset\points_ensemble_smoothed_reprojected.mat");
            labels.positions = labels.positions.points_ensemble_smoothed_reprojected; 
            labels.positions = labels.positions((371-10:521-10), :, :, :);
            labels.positions = permute(labels.positions, [3, 4, 2, 1]);
            % 
            % rows = find(~isnan(squeeze(labeled_positions(5,1,1,:))));
            % rows_n = (1:1000);
            % rows_n(rows) = [];
            % 
            % predicted_positions = load("C:\Users\amita\OneDrive\Desktop\micro-flight-lab\micro-flight-lab\Utilities\Work_W_Leap\datasets\main datasets\random frames\predicted dataset\predictions_ALL_CAMS.mat");
            % predicted_positions = predicted_positions.predictions;
            % predicted_positions = permute(predicted_positions, [3, 4, 2, 1]);
            % 
            % wings_preds_2D_x = predicted_positions(:, 1, :, :);
            % wings_preds_2D_y = predicted_positions(:, 2, :, :);
            % predicted_positions(:, 1, :, :) = wings_preds_2D_y;
            % predicted_positions(:, 2, :, :) = wings_preds_2D_x;
            % 
            % 
            % labels.positions(:, :, :, rows) = labeled_positions(:, :, :, rows);  
            % labels.positions(:, :, :, rows_n) = predicted_positions(:, :, :, rows_n);

            % ----------------------------------------------------------------------------------------

%             labels.positions = positions_wings_joints;
%             labels.positions([1,2], :,:,:) = positions_head_tail;
%             labels.positions([3,4], :,:,:) = positions_wings_joints([8,16], :,:,:);
            % Settings
            labels.config = config;

            % Timestamps
            labels.createdOn = datestr(now);
            labels.lastModified = datestr(now);

            % Initialize history
            labels.session = 1;
            addToHistory("Created labels file.");

            % Create labels file
            save(labels.savePath, '-struct', 'labels', '-v7.3')
            stocf('Created labels file: %s', labels.savePath)
        else
            % Load
            labels = load(labels.savePath);

            % Update paths
            labels.boxPath = boxPath;
            labels.savePath = repext(boxPath, '.labels.mat');

            % Update config
            if isfield(labels,'config')
                config = parse_params(labels.config,config);
            else
                labels.config = config;
            end

            if ~isfield(labels,'session')
                labels.session = 1;
            else
                labels.session = labels.session + 1;
            end

            %% change the labels from 55:1000 to NAN
%             labels.positions(: ,:, :, 56:1000) = NaN;

            stocf('Loaded existing labels file: %s', labels.savePath)
        end
        addToHistory('Started session.')

        % Convenience
        numNodes = numel(labels.skeleton.nodes);
    end

    function addToHistory(message)
    % Utility for adding a timestamped message to the history log

        session = labels.session;
        timestamp = datetime();
        message = string(message);
        historyItem = table(session, timestamp, message);
        disp(historyItem)

        if ~isfield(labels,'history') || isempty(labels.history)
            labels.history = historyItem;
        else
            labels.history = [labels.history; historyItem];
        end
    end

%% GUI
% Build GUI
global ui;
initializeGUI();
    function initializeGUI()
        ui = struct();
        
        % %%%% epipolar lines %%%%
%         ui.epiLines=zeros(6,1);
        
        % %%%% Controls figure %%%%
        ui.ctrl = struct();
        ui.ctrl.fig = figure('NumberTitle','off','MenuBar','none', ...
            'Name','LEAP Label GUI', 'WindowKeyPressFcn', @keyPress, 'DeleteFcn', @quit, ...
            'Position', config.ctrlFigPos);
        ui.ctrl.hbox = uix.HBox('Parent', ui.ctrl.fig);

        % Joints panel
        ui.ctrl.jointsPanel = uix.Panel('Parent',ui.ctrl.hbox, 'Title', 'Joints', 'Padding',5);
        ui.ctrl.jointsList = uicontrol(ui.ctrl.jointsPanel, 'Style', 'listbox', 'String', labels.skeleton.joints.name, ...
            'Callback',@(h,~,~)selectNode(h.Value));

        % Reference image
        ui.ctrl.refPanel = uix.Panel('Parent',ui.ctrl.hbox, 'Title', 'Reference', 'Padding',5);
        ui.ctrl.refAx = axes(uicontainer('Parent',ui.ctrl.refPanel));
        ui.ctrl.refImg = imagesc(labels.skeleton.refI);

        % Style
        ui.ctrl.refAx.Units = 'normalized';
        ui.ctrl.refAx.Position = [0 0 1 1];
        axis(ui.ctrl.refAx,'equal','tight','ij')
        colormap(ui.ctrl.refAx,'gray')
        noticks(ui.ctrl.refAx)
        hold(ui.ctrl.refAx,'on')

        % Plot reference skeleton
        for i = 1:size(labels.skeleton.segments,1)
            % Find default position of each nodes in the segment
            pos = labels.skeleton.pos(labels.skeleton.segments.joints_idx{i},:);

            % Plot
            plot(ui.ctrl.refAx, pos(:,1), pos(:,2), '.-', 'Color',labels.skeleton.segments.color{i}, 'LineWidth', 1);
        end

        % Draw each joint node
        ui.ctrl.refNodes = gobjects(height(labels.skeleton.joints),1);
        for i = 1:numel(ui.ctrl.refNodes)
            pos = labels.skeleton.joints.pos(i,:);
            ui.ctrl.refNodes(i) = plot(ui.ctrl.refAx, pos(1),pos(2),'o', 'Color','r');
        end

        % Set box widths
        ui.ctrl.hbox.Widths = [-1 -3];
        %%%%
        ui.img = struct();
        ui.skel = struct();
        
        color_mat=hsv(n_cams);
        channels_pc=numChannels/n_cams;
        if channels_pc == 3
            channels = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12];
        elseif channels_pc == 5
            channels = [1, 3, 5; 6, 8, 10; 11, 13, 15; 16, 18, 20];
        end

        for camInd=1:n_cams
            % %%%% Image figure %%%%
            ui.img(camInd).fig = figure('NumberTitle','off','MenuBar','none','ToolBar','none', ...
                'Name',sprintf('Frame %d/%d', config.initialFrame, numFrames), 'WindowKeyPressFcn',...
                @keyPress, 'DeleteFcn', @quit,'color',color_mat(camInd,:),...
                'Position', config.imgFigPos);
            ui.img(camInd).ax = axes(ui.img(camInd).fig);

            % display 3 times
%             img_3_times = box(:,:,(channels_pc*(camInd-1)+1):(channels_pc*(camInd-1)+3),1);
            img_3_times  = box(:, :, channels(camInd, :), 1);
            
            % display 1 time
            img_1_time = box(:,:,channels_pc*(camInd-1)+2,1);

            ui.img(camInd).img = imagesc(ui.img(camInd).ax, img_3_times);
            ui.img(camInd).img.ButtonDownFcn = @(~,~) clickImage(camInd);

            % Full figure image axes
            ui.img(camInd).ax.Units = 'normalized';
            ui.img(camInd).ax.Position = [0 0 1 1];

            % Style
            axis(ui.img(camInd).ax,'equal','tight','ij')
            colormap(ui.img(camInd).ax,'gray')
            noticks(ui.img(camInd).ax)
            hold(ui.img(camInd).ax,'on')

            % Initialize skeleton drawing container
            ui.skel(camInd).segs = [];
            ui.skel(camInd).nodes = [];
            %%%%
        end

        % %%%% Status figure %%%%
        % Initialize status container
        ui.status = struct();
        ui.status.selectedNode = [];
        ui.status.movedNodes = false(numNodes,n_cams);
        ui.status.currentFrame = config.initialFrame;
        ui.status.unsavedChanges = false(numFrames,1);
        ui.status.initialPos = [];

        % Get full status indicators for all frames
        status = getStatus();
        numInitialized = sum(all(status == 1,1));
        numLabeled = sum(all(status == 2,1));

        % Create figure window
        ui.status.fig = figure('NumberTitle','off','MenuBar','none','ToolBar','none', ...
            'Name',sprintf('Status: %d/%d (%.2f%%) labeled', numLabeled, numFrames, numLabeled/numFrames*100), ...
            'WindowKeyPressFcn', @keyPress, 'DeleteFcn', @quit);
%             'Position', config.statusFigPos);
        ui.status.hbox = uix.HBox('Parent', ui.status.fig, 'Padding',3);

        % Status panel (left)
        ui.status.statusPanel = uix.Panel('Parent',ui.status.hbox, 'Title','Status', 'Padding',5);
        ui.status.statusBoxes = uix.VBox('Parent', ui.status.statusPanel);

        % Status text
        ui.status.stats = uix.VBox('Parent',ui.status.statusBoxes);

        ui.status.framesInitialized = uicontrol(ui.status.stats,'Style','text','HorizontalAlignment','left',...
            'String',sprintf('Initialized: %d/%d (%.3f%%)', numInitialized, numFrames, numInitialized/numFrames*100));
        ui.status.framesLabeled = uicontrol(ui.status.stats,'Style','text','HorizontalAlignment','left',...
            'String',sprintf('Labeled: %d/%d (%.3f%%)', numLabeled, numFrames, numLabeled/numFrames*100));
        ui.status.stats.Heights = ones(1, numel(ui.status.stats.Children)) * 15;

        % Status bars
        ui.status.fullAx = axes(uicontainer('Parent',ui.status.statusBoxes));
        ui.status.fullImg = imagesc(ui.status.fullAx, 1:numFrames, 1:numNodes, status, 'ButtonDownFcn', @clickStatusbar);
        axis(ui.status.fullAx,'tight','ij')
        hold(ui.status.fullAx,'on');
        zoomBoxIdx = zoomBoxWindow + ui.status.currentFrame;
        zoomBoxPts = [
            zoomBoxIdx(1) 0
            zoomBoxIdx(end) 0
            zoomBoxIdx(end) numNodes
            zoomBoxIdx(1) numNodes
            zoomBoxIdx(1) 0
            ];
        ui.status.fullZoomBox = patch(ui.status.fullAx, zoomBoxPts(:,1),zoomBoxPts(:,2),'w','PickableParts','none');
        ui.status.fullZoomBox.FaceAlpha = 0.25;
        ui.status.fullZoomBox.EdgeColor = 'w';
        colormap(ui.status.fullAx, statusCmap)
        caxis(ui.status.fullAx,[0 2])
        ui.status.fullAx.XLim = [-0.5 0.5] + [1 numFrames];
        ui.status.fullAx.YLim = [-0.5 0.5] + [1 numNodes];
%         ui.status.fullAx.YTick = 1:numNodes;
%         ui.status.fullAx.YTickLabel = labels.skeleton.nodes;
%         ui.status.fullAx.YAxis.TickLabelInterpreter = 'none';

        % Status bars (zoomed)
        ui.status.zoomAx = axes(uicontainer('Parent',ui.status.statusBoxes));
        ui.status.zoomImg = imagesc(ui.status.zoomAx, zoomBoxIdx, 1:numNodes, zeros(numNodes,numel(zoomBoxIdx)), 'ButtonDownFcn', @clickStatusbar);
        axis(ui.status.zoomAx,'tight','ij')
        colormap(ui.status.zoomAx, statusCmap)
        caxis(ui.status.zoomAx,[0 2])
        ui.status.zoomAx.YLim = [-0.5 0.5] + [1 numNodes];
%         ui.status.zoomAx.YTick = 1:numNodes;
%         ui.status.zoomAx.YTickLabel = labels.skeleton.nodes;
%         ui.status.zoomAx.YAxis.TickLabelInterpreter = 'none';

        % Set UI heights
        ui.status.statusBoxes.Heights = [sum(ui.status.stats.Heights)+5 -1 -1];

        % Settings panel (right)
        ui.status.configPanel = uix.Panel('Parent',ui.status.hbox, 'Title','Settings','Padding',5);
        ui.status.configButtons = uix.VBox('Parent',ui.status.configPanel);

        % Auto-save
        uicontrol(ui.status.configButtons,'Style','checkbox','Value',config.autoSave, ...
            'Callback',@(h,~)setConfig('autoSave',h.Value), ...
            'String','Autosave labels','TooltipString','Automatically saves changes to disk when changing frames or exiting.');

        % Shuffle frame order
        uicontrol(ui.status.configButtons,'Style','checkbox','Value',config.shuffleFrames, ...
            'Callback',@(h,~)setConfig('shuffleFrames',h.Value), ...
            'String','Shuffle frame order','TooltipString','Shuffled order is fixed within this session. Uncheck to use file ordering.');

        % Click nearest
        uicontrol(ui.status.configButtons,'Style','checkbox','Value',config.clickNearest, ...
            'Callback',@(h,~)setConfig('clickNearest',h.Value), ...
            'String','Click to move nearest joint','TooltipString','If unchecked, clicking on the image moves the currently selected joint.');

        % Draggable markers
        uicontrol(ui.status.configButtons,'Style','checkbox','Value',config.draggable, ...
            'Callback', @(h,~)toggleDraggableMarkers(h.Value), ...
            'String','Draggable markers','TooltipString','If unchecked, joint markers can only be moved by clicking or keyboard.');

        % Alt + arrows to move nodes
        uicontrol(ui.status.configButtons,'Style','checkbox','Value',config.altArrowsToMoveNodes, ...
            'Callback', @(h,~)setConfig('altArrowsToMoveNodes',h.Value), ...
            'String','Alt + arrow keys move markers','TooltipString','If unchecked, move markers with Alt + arrow keys, change frames with arrow keys.');

        % Export confidence maps
        uicontrol(ui.status.configButtons,'Style','pushbutton', ...
            'Callback', @(h,~)generateTrainingSet(), ...
            'String','Generate training set','TooltipString','Creates a test set with confidence maps for training a network.');

        % Fast training
        uicontrol(ui.status.configButtons,'Style','pushbutton', ...
            'Callback', @(h,~)fastTrain(), ...
            'String','Fast train network','TooltipString','Trains a network for initialization using fast presets.');

        % Initialization from predictions
        uicontrol(ui.status.configButtons,'Style','pushbutton', ...
            'Callback', @(h,~)predictInitializations(), ...
            'String','Initialize with trained model','TooltipString','Generates predictions for all frames and uses it as initialization.');


        % Set UI sizes
        ui.status.configButtons.Heights = ones(1, numel(ui.status.configButtons.Children)) * 25;
        ui.status.hbox.Widths = [-1 175];

        % Give focus back to main image window
        figure(ui.img(1).fig);
    end
    function toggleDraggableMarkers(TF)
    % Sets whether joint markers are draggable using the mouse

        if TF
            for camInd=1:n_cams
                set(ui.skel(camInd).nodes, 'PickableParts','visible');
                draggable(ui.skel(camInd).nodes,@nodesMoved, 'endFcn',@nodesMoveEnd);
            end
        else
            for camInd=1:n_cams
                draggable(ui.skel(camInd).nodes, 'off');
                set(ui.skel(camInd).nodes,'PickableParts','none');
            end
        end
        setConfig('draggable',TF);
    end
    function setConfig(configField, value)
    % Helper to set config fields to specified value
        config.(configField) = value;
    end

    function quit(h,~)
    % Quit callback to close all windows simultaneously
        % Log to history
        addToHistory("Finished session.")

        % Save
        if config.autoSave && isequal(h, ui.img.fig)
            saveLabels();
        end

        % Delete figs
        for camInd=1:n_cams
            delete(ui.img(camInd).fig)
        end
        delete(ui.ctrl.fig)
        delete(ui.status.fig)
    end

    function keyPress(~,evt)
    % Hotkeys
        % exclusive modifier flags:
        noModifier = isempty(evt.Modifier);
        shiftOnly = isequal(evt.Modifier, {'shift'});
        ctrlOnly = isequal(evt.Modifier, {'control'});
        altOnly = isequal(evt.Modifier, {'alt'});

        % non-exclusive:
        altPressed = ismember({'alt'}, evt.Modifier);
        ctrlPressed = ismember({'control'}, evt.Modifier);
        shiftPressed = ismember({'shift'}, evt.Modifier);
        evt.Key
        switch evt.Key
            case 'q'
                delete(ui.img.fig)
            case 's'
                saveLabels()
            case 'r'
                if noModifier % current node
                    resetNodes(ui.status.selectedNode);
                elseif shiftOnly % all nodes
                    resetNodes();
                end
            case 'd'
                if noModifier % current node
                    setNodesToDefault(ui.status.selectedNode);
                elseif shiftOnly % all nodes
                    setNodesToDefault();
                end
            case 'tab'
                if noModifier
                    selectNode(mod(ui.status.selectedNode-1+1, numNodes) + 1);
                elseif shiftOnly
                    selectNode(mod(ui.status.selectedNode-1-1, numNodes) + 1);
                end
            case 'downarrow'
                dXY = [0 1];
                if (config.altArrowsToMoveNodes && altPressed) || ~config.altArrowsToMoveNodes
                    if noModifier
                        nudgeNode(dXY)
                    elseif shiftOnly
                        nudgeNode(dXY * 5)
                    elseif ctrlOnly
                        nudgeSegment(dXY)
                    end
                end
            case 'uparrow'
                dXY = [0 -1];
                if (config.altArrowsToMoveNodes && altPressed) || ~config.altArrowsToMoveNodes
                    if noModifier
                        nudgeNode(dXY)
                    elseif shiftOnly
                        nudgeNode(dXY * 5)
                    elseif ctrlOnly
                        nudgeSegment(dXY)
                    end
                end
            case 'leftarrow'
                if (config.altArrowsToMoveNodes && altPressed) || (~config.altArrowsToMoveNodes && ~altPressed)
                    dXY = [-1 0] - (shiftPressed * 4);
                    if ctrlPressed; nudgeSegment(dXY);
                    else; nudgeNode(dXY); end
                else
                    dt = -1 - (shiftPressed * 4);
                    if config.shuffleFrames
                        idx = find(shuffleIdx == ui.status.currentFrame);
                        goToFrame(shuffleIdx(mod(idx-1+dt, numFrames) + 1))
                    else
                        goToFrame(mod(ui.status.currentFrame-1+dt, numFrames) + 1)
                    end
                end
            case 'rightarrow'
                if (config.altArrowsToMoveNodes && altPressed) || (~config.altArrowsToMoveNodes && ~altPressed)
                    dXY = [1 0] + (shiftPressed * 4);
                    if ctrlPressed; nudgeSegment(dXY);
                    else; nudgeNode(dXY); end
                else
                    dt = 1 + (shiftPressed * 4);
                    if config.shuffleFrames
                        idx = find(shuffleIdx == ui.status.currentFrame);
                        goToFrame(shuffleIdx(mod(idx-1+dt, numFrames) + 1))
                    else
                        goToFrame(mod(ui.status.currentFrame-1+dt, numFrames) + 1)
                    end
                end
            case 'space'
                % Get labeling status for all frames
                labeled = getStatus() == 2;

                % Consider current joint only if shift is pressed
                if shiftPressed; labeled = labeled(ui.status.selectedNode,:);
                else; labeled = all(labeled,1); end

                % Find unlabeled frames excluding current frame
                unlabeledIdxs = setdiff(find(labeled), ui.status.currentFrame);

                if ~isempty(unlabeledIdxs)
                    if ctrlPressed
                        % Go to random unlabeled frame
                        goToFrame(datasample(unlabeledIdxs,1));
                    else
                        % Go to first unlabeled frame
                        goToFrame(unlabeledIdxs(1))
                    end

                end
            case 'g'
                % go to frame dialog
%                 if ctrlOnly
                % TODO:
                %   - custom dialog box that starts focused on the textbox
                %     and returns after pressing Enter/Esc
                answer = inputdlg('Skip to frame index:','Skip to frame',1,{num2str(ui.status.currentFrame)});
                try
                    idx = round(str2double(answer));
                    if idx >= 1 && idx <= numFrames
                        goToFrame(idx);
                    end
                catch
                end
%                 end
            case 'f'
                markAllCorrect();
            case 'l'
                EpipolarIntersect();
            otherwise
%                 evt
        end
    end

    function clickImage(camInd)
    % Callback to image clicks (but not on nodes)
        % Pull out clicked point coordinate
        pt = ui.img(camInd).ax.CurrentPoint(1,1:2);

        % Get current node positions
        pos = getNodePositions();

        if config.clickNearest
            % Find nearest node location
            i = argmin(rownorm(pos - pt));
        else
            % Use current selection
            i = ui.status.selectedNode;
        end

        % Update node position
        pos(i,:,camInd) = pt;
        updateSkeleton(pos,camInd);

    end

    function clickStatusbar(h,evt)
    % Callback for seeking via mouse-click on the status bars
        if evt.Button == 1
            idx = clip(round(evt.IntersectionPoint(1)),[1 numFrames]);
            goToFrame(idx);
        end
    end

    function status = getStatus(idx)
    % Utility function that checks labels for completeness status
    % Returns [numJoints x numel(idx)] matrix with values:
    %   0: default
    %   1: initialized
    %   2: labeled

        % Get status for all frames by default
        if nargin < 1; idx = 1:numFrames; end

        % Initialize as default (0)
        status = zeros(numNodes, numel(idx));

        % Check for initialization
        isInitialized = squeeze(any(all(~isnan(labels.initialization(:,:,:,idx)),2),3));
        status(isInitialized) = 1;

        % Check for user labels
        isLabeled = squeeze(any(all(~isnan(labels.positions(:,:,:,idx)),2),3));
        status(isLabeled) = 2;
    end

%% Training and dataset generation
    function predictInitializations(modelPath)
    % Generates predictions for the entire dataset and uses those for
    % initialization of unlabeled frames.

%         if nargin < 1 || isempty(modelPath)
%             modelPath = uibrowse([],[],'Select model folder...', 'dir');
%             if isempty(modelPath) || ~exists(modelPath); return; end
%         end

        % TODO: better system for choosing final vs best validation model
%         if exists(ff(modelPath, 'final_model.h5'))
%             numValidationSamples = numel(loadvar(ff(modelPath,'training_info.mat'),'val_idx'));
% %             numWeights = numel(dir_files(ff(modelPath,'weights')));
%             if numValidationSamples < 500
%                 modelPath = ff(modelPath,'final_model.h5');
%             end
%         end

%         numValidationSamples = numel(loadvar(ff(modelPath,'training_info.mat'),'val_idx'));
%         if exists(ff(modelPath, 'best_model.h5')) && numValidationSamples > 500
%             modelPath = ff(modelPath, 'best_model.h5');
%         else
%             modelPath = ff(modelPath, 'best_model.h5');
% %             modelPath = ff(modelPath, 'final_model.h5');
%         end
        
%         !!!!!! ADD questdlg for 3d or singles !!!!!!
        
        %%%% noamler (good for singles)
%         boxSinglesPath=Dataset3d2single(boxPath);
%         preds = predict_box(boxSinglesPath, modelPath, false);
%         delete(boxSinglesPath)
        %%%
        
        % from file
        modelPath='look in readme';
        preds = h5readgroup(predictions_xx);
        preds_1 = load(predictions_path);
        preds_1 = preds_1.predictions_from_3D_to_2D;
        preds_1 = preds_1(:,:, 1:16,:);
        preds_1 = reshape(preds_1, [2000, 16,2]);
        preds_1 = permute(preds_1, [2,3,1]);
        preds.positions_pred = single(preds_1) + 1;
        
%         num_frames=size(preds.positions_pred,3)/2;
%         preds.positions_pred=cat(1,preds.positions_pred(1:2,:,1:num_frames),...
%             preds.positions_pred(1:2,:,(1:num_frames)+num_frames),...
%             preds.positions_pred(3:6,:,1:num_frames),...
%             preds.positions_pred(3:6,:,(1:num_frames)+num_frames),...
%             preds.positions_pred(7,:,1:num_frames),...
%             preds.positions_pred(7,:,(1:num_frames)+num_frames),...
%             preds.positions_pred((1:2)+7,:,1:num_frames),...
%             preds.positions_pred((1:2)+7,:,(1:num_frames)+num_frames),...
%             preds.positions_pred((3:6)+7,:,1:num_frames),...
%             preds.positions_pred((3:6)+7,:,(1:num_frames)+num_frames),...
%             preds.positions_pred((7)+7,:,1:num_frames),...
%             preds.positions_pred((7)+7,:,(1:num_frames)+num_frames),...
%             preds.positions_pred((1:2)+7*2,:,1:num_frames),...
%             preds.positions_pred((1:2)+7*2,:,(1:num_frames)+num_frames),...
%             preds.positions_pred((3:6)+7*2,:,1:num_frames),...
%             preds.positions_pred((3:6)+7*2,:,(1:num_frames)+num_frames),...
%             preds.positions_pred((7)+7*2,:,1:num_frames),...
%             preds.positions_pred((7)+7*2,:,(1:num_frames)+num_frames));
        % Predict (good for 3d)
%         preds = predict_box(boxPath, modelPath, false);
        
        % Save
%         labels.initialization = preds.positions_pred;
        
        %singles
%         size3d=size(preds.positions_pred,3)/3;
%         labels.initialization=permute(cat(4,preds.positions_pred(:,:,1:size3d),...
%             preds.positions_pred(:,:,(1:size3d)+size3d),...
%             preds.positions_pred(:,:,(1:size3d)+2*size3d)),[1,2,4,3]);
        %3d
%         labels.initialization=permute(reshape(preds.positions_pred,size(preds.positions_pred,1)/3,3,2,[]),[1,3,2,4]);
        
        labels.initialization = NaN(numel(labels.skeleton.nodes), 2,n_cams, numFrames, 'single');
        szz=size(labels.initialization);
%         labels.initialization=reshape(preds.positions_pred,szz(1),szz(2),4,[]);
        labels.initialization=permute(cat(4,preds.positions_pred(:,:,1:szz(4)),...
            preds.positions_pred(:,:,(szz(4)+1):(2*szz(4))),...
            preds.positions_pred(:,:,(2*szz(4)+1):(3*szz(4))),...
            preds.positions_pred(:,:,(3*szz(4)+1):(4*szz(4)))),[1,2,4,3]);


        saveLabels();

        % Update status
        isInitialized = squeeze(all(~isnan(labels.initialization),2));
        numInitialized = sum(all(isInitialized,1));
        ui.status.framesInitialized.String = sprintf('Initialized: %d/%d (%.2f%%)', numInitialized, numFrames, numInitialized/numFrames*100);

        % Update status bars
        status = getStatus();
        ui.status.fullImg.CData = status;
        zoom_idx = ui.status.zoomImg.XData > 0 & ui.status.zoomImg.XData <= size(status,2);
        ui.status.zoomImg.CData(:,zoom_idx) = status(:,ui.status.zoomImg.XData(zoom_idx));

        % Log event
        addToHistory(['Initialized with model: ' modelPath])

        % Calculate error rate on labels
        labeled = all(getStatus() == 2,1);
        pos_gt = labels.positions(:,:,labeled);
        pos_pred = labels.initialization(:,:,labeled);
        pred_metrics = compute_errors(pos_pred,pos_gt);

        % Display errors
        printf('Error: mean = %.2f, s.d. = %.2f', mean(pred_metrics.euclidean(:)), std(pred_metrics.euclidean(:)))
        prcs = [50 75 90];
        prc_errs = prctile(pred_metrics.euclidean(:), prcs);
        for i = 1:numel(prcs)
            printf('       %d%% = %.3f', prcs(i), prc_errs(i))
        end

        % Replot
        goToFrame(ui.status.currentFrame);
    end
    function generateTrainingSet()
        
        % Default save path
        defaultSavePath = ff(fileparts(boxPath), 'training', [get_filename(boxPath,true) '.h5']);
        defaultSavePath = get_new_filename(defaultSavePath,true);
        
        % Create dialog with parameters
        [params, buttonPressed] = settingsdlg(...
            'WindowWidth', 400,...
            'title','Generate a training set', ...
            'Description','Export a dataset for training based on the current labels.',...
            'separator','General options',...
            {'Save path';'savePath'}, defaultSavePath, ...
            {'Scale - for resizing images';'scale'}, 1, ...
            {'Sigma - kernel size for confidence maps';'sigma'}, 3, ...
            {'Test set fraction - held out frames';'testFraction'},0,...
            {'Shuffle - randomize saved dataset order';'postShuffle'}, true, ...
            {'Compress - reduce file size, but slower to load';'compress'}, true, ...
            'separator','Data mirroring',...
            {'Mirror images - augment by flipping along the body axis';'mirroring'}, [true, false], ...
            {'Animal orientation';'animalOrientation'},  {'left/right','top/bottom'} ...
            );
        
        % Cancel if OK was not pressed (cancel or window closed)
        if ~strcmpi(buttonPressed,'ok'); return; end
        
        % Convert listbox input to boolean for orientation
        params.horizontalOrientation = strcmpi(params.animalOrientation,'left/right');
        
        % Check for existing save path
        if exists(params.savePath)
            answer = questdlg('Save path already exists, overwrite existing file?', 'Overwrite file', 'Overwrite', 'Cancel', 'Overwrite');
            if ~strcmpi(answer, 'Overwrite'); return; end
        end
        
        % Run!
        % noamler
%         generate_training_set_3d(boxPath,params);
%         generate_training_set_3d_to_singles(boxPath,params);
%         generate_training_set_3d_amitai(cropZone ,boxPath, params);
        generate_training_set_no_masks(cropZone ,boxPath, easyWandData, params);
        % Log action
        addToHistory('Generated training set.')
    end
    function fastTrain()
        % Generate a training set for fast training from current labels
        
        % Build default output path
        runName = sprintf('%s-n=%d', datestr(now,'yymmdd_HHMMSS'), numLabeled);
        defaultModelsFolder = ff(fileparts(boxPath), 'models');
        
        % Create dialog with parameters
        [params, buttonPressed] = settingsdlg(...
            'WindowWidth', 500,...
            'title','Fast training', ...
            'Description','Quickly train a model using current labels and predict on remaining frames as initialization.',...
            'separator','Dataset',...
            {'Scale - for resizing images';'scale'}, 1, ...
            {'Sigma - kernel size for confidence maps';'sigma'}, 5, ...
            'separator','Data mirroring',...
            {'Mirror images - augment by flipping along the body axis';'mirroring'}, [true, false], ...
            {'Animal orientation';'animalOrientation'}, {'left/right','top/bottom'}, ...
            'separator','Model',...
            {'Network architecture';'netName'},{'leap_cnn','hourglass','stacked_hourglass'},...
            {'Filters - base number of filters for model';'filters'},32,...
            {'Upsampling layers - use bilinear upsampling instead of transposed conv';'upsamplingLayers'},true,...
            'separator','Training',...
            {'Model path - folder to save run data to';'modelsFolder'},defaultModelsFolder,...
            {'Rotate angle - augment data via random rotations';'rotateAngle'},5,...
            {'Validation set fraction - frames used for validation';'valSize'},0,...
            {'Epochs - number of rounds of training';'epochs'},15,...
            {'Batch size - number of samples per batch';'batchSize'},50,...
            {'Batches per epoch - number of batches of samples per round';'batchesPerEpoch'},50,...
            {'Validation batches per epoch - number of batches to use for validation';'valBatchesPerEpoch'},10,...
            {'Save every epoch - save weights from every epoch instead of just best+final';'saveEveryEpoch'},false,...
            'separator','Training (advanced)',...
            {'Reduce LR factor - drop learning rate when loss plateaus';'reduceLRFactor'},0.1,...
            {'Reduce LR patience - wait after loss plateaus before reducing LR';'reduceLRPatience'},2,...
            {'Reduce LR cooldown - wait after reducing LR before detecting plateau';'reduceLRCooldown'},0,...
            {'Reduce LR min delta - minimum change in loss to not plateau';'reduceLRMinDelta'},1e-5,...
            {'Reduce LR min LR - minimum LR to not drop below';'reduceLRMinLR'},1e-10,...
            {'AMSGrad - optimizer variant for more emphasis on rare data';'amsgrad'},true ...
            );
        
        % Cancel if OK was not pressed (cancel or window closed)
        if ~strcmpi(buttonPressed,'ok'); return; end
        
        % Convert listbox input to boolean for orientation
        params.horizontalOrientation = strcmpi(params.animalOrientation,'left/right');
        
        % Generate temporary training set file
        dataPath = [tempname '.h5'];
        dataPath = generate_training_set_3d(boxPath,'savePath',dataPath,...
            'scale',params.scale,...
            'mirroring',params.mirroring,...
            'horizontalOrientation',params.horizontalOrientation,...
            'sigma',params.sigma, ...
            'normalizeConfmaps',true,...
            'postShuffle',true, ...
            'testFraction',0);
        
        % Log action
        addToHistory(sprintf('Fast training (n = %d)', numLabeled))
        
        % Create CLI command for training
        basePath = fileparts(funpath(true));
        
%         !!!NOAMLER; add this to activate virtualenv before!!!!!!
%         system('cd C:\git reps\LEAPvenv\Scripts & activate & python -V','-echo')
        
        cmd = {'cd C:\git reps\LEAPvenv\Scripts & activate &'
            'python'
            ['"' ff(basePath, 'leap\training.py') '"']
            ['"' dataPath '"']
            ['--base-output-path="' params.modelsFolder '"']
            ['--run-name="' runName '"']
            ['--net-name="' params.netName '"']
            sprintf('--filters=%d',params.filters)
            sprintf('--rotate-angle=%d', params.rotateAngle)
            sprintf('--val-size=%.5f', params.valSize)
            sprintf('--epochs=%d', params.epochs)
            sprintf('--batch-size=%d', params.batchSize)
            sprintf('--batches-per-epoch=%d', params.batchesPerEpoch)
            sprintf('--val-batches-per-epoch=%d', params.valBatchesPerEpoch)
            sprintf('--reduce-lr-factor=%.10f', params.reduceLRFactor)
            sprintf('--reduce-lr-patience=%d', params.reduceLRPatience)
            sprintf('--reduce-lr-cooldown=%d', params.reduceLRCooldown)
            sprintf('--reduce-lr-min-delta=%.10f', params.reduceLRMinDelta)
            sprintf('--reduce-lr-min-lr=%.10f', params.reduceLRMinLR)
            };
        
        if params.upsamplingLayers; cmd{end+1} = '--upsampling-layers'; end
        if params.saveEveryEpoch; cmd{end+1} = '--save-every-epoch'; end
        if params.amsgrad; cmd{end+1} = '--amsgrad'; end
        
        cmd = strjoin(cmd);
        disp(cmd)

        % Train!
        try
            exit_code = system(cmd);
%             [exit_code,cmd_output] = system(cmd);
        catch ME
            delete(dataPath)
            rethrow(ME)
        end
        delete(dataPath)

        % TODO: parse this out from python output?
        modelPath = ff(params.modelsFolder, runName);

        % Run trained model on data to initialize labels
        if exists(ff(modelPath, 'final_model.h5'))
            predictInitializations(modelPath)
        end
    end

%% Ploting
initializeSkeleton();

    function initializeSkeleton()
    % Creates graphics objects representing the interactive skeleton
        for camInd=1:length(ui.img)
            % Draw each line segment
            if ~isempty(ui.skel(camInd).segs); delete(ui.skel(camInd).segs); end
            ui.skel(camInd).segs = gobjects(size(labels.skeleton.segments,1),1);
            for i = 1:numel(ui.skel(camInd).segs)
                % Find default position of each nodes in the segment
                pos = labels.skeleton.pos(labels.skeleton.segments.joints_idx{i},:);

                % Plot
                ui.skel(camInd).segs(i) = plot(ui.img(camInd).ax, pos(:,1), pos(:,2), '.-', ...
                    'Color',labels.skeleton.segments.color{i});

                % Add metadata
                ui.skel(camInd).segs(i).UserData.seg_idx = i;
                ui.skel(camInd).segs(i).UserData.seg_joints_idx = labels.skeleton.segments.joints_idx{i};
            end

            % Clicks on the skeleton edges should pass through to the image
            set(ui.skel(camInd).segs, 'PickableParts', 'none');

            % Draw each joint node
            if ~isempty(ui.skel(camInd).nodes); delete(ui.skel(camInd).nodes); end
    %         status = getStatus(ui.status.currentFrame); statusCmap(status(i)+1,:)
            ui.skel(camInd).nodes = gobjects(height(labels.skeleton.joints),1);
            for i = 1:numel(ui.skel(camInd).nodes)
                ui.skel(camInd).nodes(i) = plot(ui.img(camInd).ax,labels.skeleton.joints.pos(i,1),labels.skeleton.joints.pos(i,2),'o',...
                    'Color','w', 'LineWidth', 1, 'PickableParts','none');
                ui.skel(camInd).nodes(i).UserData.node_idx = i;
            end

            % Make movable and add callbacks
            if config.draggable
                set(ui.skel(camInd).nodes, 'PickableParts','visible');
                draggable(ui.skel(camInd).nodes,@nodesMoved,'endFcn',@nodesMoveEnd);
            end
        end
    end
    function pos = getNodePositions()
    % Utility function that returns node positions from the corresponding graphics objects
        pos= NaN(numel(ui.skel(1).nodes),2,n_cams);
        for camInd=1:n_cams
            for i = 1:numel(ui.skel(camInd).nodes)
                pos(i,:,camInd) = [ui.skel(camInd).nodes(i).XData ui.skel(camInd).nodes(i).YData];
            end
        end
    end
    function updateSkeleton(pos,camInd_pressed)
    % Updates pre-initialiazed skeleton graphics objects
    
        if nargin < 2
            camInd_pressed=1:n_cams;
        end
        if nargin < 1
            % Get current node positions from graphics
            pos = getNodePositions();
        else
            % Update node positions
            for camInd=camInd_pressed
                for i = 1:size(pos,1)
                    % Check for modification to graphics positions
                    old_pos = [ui.skel(camInd).nodes(i).XData ui.skel(camInd).nodes(i).YData];
                    if ~isequal(pos(i,:,camInd), old_pos)
                        % Update graphics
                        ui.skel(camInd).nodes(i).XData = pos(i,1,camInd);
                        ui.skel(camInd).nodes(i).YData = pos(i,2,camInd);
                    end
                end
               
            end
        end

        % Check for changes
        for i = 1:numNodes
            if ~isequal(pos(i,:), ui.status.initialPos(i,:))
                % Mark node as moved
                ui.status.movedNodes(i,camInd_pressed) = true;

                % Denote unsaved changes
                ui.status.unsavedChanges(ui.status.currentFrame) = true;
            end
        end
        
        color_mat=hsv(n_cams);
        % Set defaults
        for camInd=camInd_pressed
            set(ui.skel(camInd).nodes, 'Marker', 'o'); % Default marker (no changes)
            set(ui.skel(camInd).nodes, 'MarkerSize', config.nodeSize); % Default size (unselected)

            % Update epipolar lines
            for proj_ind=find(coupless(:,1)==camInd)'
                try
                    delete(ui.epiLines(proj_ind))
                end
                
                cz_this=double(cropZone(:,coupless(proj_ind,1),ui.status.currentFrame));
                line=fundMats(:,:,proj_ind)*...
                    [(cz_this(2)+pos(ui.status.selectedNode,1,camInd));...
                    (im_height+1-(cz_this(1)+pos(ui.status.selectedNode,2,camInd)));...
                    1];
                line=-line./line(2,:);
                cz_other=double(cropZone(:,coupless(proj_ind,2),ui.status.currentFrame));                  

                ui.epiLines(proj_ind)=fplot(ui.img(coupless(proj_ind,2)).ax,...
                   @(x) im_height+1-cz_other(1)-(line(1)*(x+cz_other(2))+line(3)),'Color',color_mat(camInd,:));
                set(ui.epiLines(proj_ind), 'PickableParts', 'none');
                
                ui.img(coupless(proj_ind,2)).ax.XLim=[0,frame_size];
                ui.img(coupless(proj_ind,2)).ax.YLim=[0,frame_size];
            end
        end
        set(ui.ctrl.refNodes, 'MarkerSize', config.nodeSize); % Default size (unselected)

        % Update node colors based on status
        status = getStatus(ui.status.currentFrame);
        for i = 1:numNodes
            % Set status color
            for camInd=camInd_pressed
                ui.skel(camInd).nodes(i).Color = statusCmap(status(i)+1,:);

                % Uncommitted changes
                camInd_pressed
                if ui.status.movedNodes(i,camInd_pressed); ui.skel(camInd).nodes(i).Marker = 's'; end

                % Selected node
                if ui.status.selectedNode == i
                    ui.skel(camInd).nodes(i).MarkerSize = 9;
                    ui.ctrl.refNodes(i).MarkerSize = 9;
                    ui.skel(camInd).nodes(i).Color=[0,0,1];
                end
                
                fontsize=20;
                if i==13
                    try
                        delete(ui.wingText(1,camInd))
                    catch
                    end
                    ui.wingText(1,camInd)=text(ui.img(camInd).ax,pos(i,1,camInd),...
                        pos(i,2,camInd), 'L', 'FontSize', fontsize, 'Color', [0,1,0],...
                        'FontWeight','bold','PickableParts','none');
                    uistack(ui.wingText(1,camInd),'bottom')
                    uistack(ui.wingText(1,camInd),'up')
                end
                if i==14
                    try
                        delete(ui.wingText(2,camInd))
                    catch
                    end
                    ui.wingText(2,camInd)=text(ui.img(camInd).ax,pos(i,1,camInd),...
                        pos(i,2,camInd), 'R', 'FontSize', fontsize, 'Color', [1,0,0],...
                        'FontWeight','bold','PickableParts','none');  
                    uistack(ui.wingText(2,camInd),'bottom')
                    uistack(ui.wingText(2,camInd),'up')
                end
            end
        end

        % Update edges
        for camInd=1:n_cams
            for i = 1:numel(ui.skel(camInd).segs)
                ui.skel(camInd).segs(i).XData(:) = pos(ui.skel(camInd).segs(i).UserData.seg_joints_idx,1,camInd);
                ui.skel(camInd).segs(i).YData(:) = pos(ui.skel(camInd).segs(i).UserData.seg_joints_idx,2,camInd);
            end
        end

        drawnow;
    end

    function nodesMoved(h)
    % Called while node is being moved to update skeleton
        
        % Get node index
        node_idx = h.UserData.node_idx;

        % Set selected node
        if ui.status.selectedNode ~= node_idx
            selectNode(node_idx)
        end
        
        touchedFig=gcf;
        camInd=touchedFig.Number-1;
        pt = ui.img(camInd).ax.CurrentPoint(1,1:2);
        
        % Get current node positions
        pos = getNodePositions();

        if config.clickNearest
            % Find nearest node location
            i = argmin(rownorm(pos - pt));
        else
            % Use current selection
            i = ui.status.selectedNode;
        end

        % Update node position
        pos(i,:,camInd) = pt;
        updateSkeleton(pos,camInd);
    
    
        
% 
%         % Update
%         updateSkeleton()

    end

    function nodesMoveEnd(h)
    % Called when the node is released after moving
        % Get node index
        node_idx = h.UserData.node_idx;

        % Set selected node
        if ui.status.selectedNode ~= node_idx
            selectNode(node_idx)
        end
        
        touchedFig=gcf;
        camInd=touchedFig.Number-1;

        % Update
        pos = getNodePositions();
        updateSkeleton(pos,camInd)

    end

    function selectNode(i)
    % Utility function that sets the selected node across the entire GUI

        % Check for changes
        previousSelection = ui.status.selectedNode;

        if ~isequal(previousSelection, i)
            % Set selected node
            ui.status.selectedNode = i;

            % Update listbox
            ui.ctrl.jointsList.Value = i;

            % Update graphics
            updateSkeleton();
        end
    end

    function nudgeNode(dXY, i)
    % Utility function for moving a node by a delta amount
        if nargin < 2; i = ui.status.selectedNode; end

        % Get and update node position
        pos = getNodePositions();
        pos(i,:) = pos(i,:) + dXY;

        % Update
        updateSkeleton(pos);
    end

    function nudgeSegment(dXY, i)
    % Utility function for moving all segments with a node by a delta amount
        if nargin < 2; i = ui.status.selectedNode; end

        % Find each segment with the current node and pull out all nodes
        seg_nodes = {};
        for j = 1:height(labels.skeleton.segments)
            idx = labels.skeleton.segments.joints_idx{j};
            if any(idx == i)
                seg_nodes{end+1} = idx;
            end
        end

        % Get the union of the set to make sure we don't double move any nodes
        seg_nodes = unique(cellcat(seg_nodes));

        % Get current positions
        pos = getNodePositions();

        % Move all nodes
        for j = 1:numel(seg_nodes)
            pos(j,:) = pos(j,:) + dXY;
        end

        % Update edges
         updateSkeleton(pos);
    end

    function setNodesToDefault(node_idx)
    % Utility function to reset nodes to default position from the skeleton template
        if nargin < 1; node_idx = 1:numNodes; end

        % Get current positions
        pos = getNodePositions();

        % Get default positions
        default_pos = labels.skeleton.joints.pos;

        % Update with defaults
%         pos(node_idx,:) = default_pos(node_idx,:);
        labels.positions(node_idx,:,ui.status.currentFrame) = NaN;
        ui.status.movedNodes(node_idx,:) = false;

        % Update
        updateSkeleton();
    end

    function pos = getInitialPos(idx)
    % Utility to compute the initial node positions for a single frame
        if nargin < 1; idx = ui.status.currentFrame; end

        % Start off with defaults
        pos = repmat(labels.skeleton.joints.pos,[1,1,n_cams]);
        
        % Update with initialized positions
        init_pos = labels.initialization(:,:,:,idx);
        init_nodes = find(all(all(~isnan(init_pos),2),3));
        pos(init_nodes,:,:) = init_pos(init_nodes,:,:);

        % Update with user-labeled positions
        label_pos = labels.positions(:,:,:,idx);
        label_nodes = find(all(all(~isnan(label_pos),2),3));
        pos(label_nodes,:,:) = label_pos(label_nodes,:,:);
    end

    function resetNodes(node_idx)
    % Utility function to reset nodes to their initial positions when the frame was drawn
        if nargin < 1; node_idx = 1:numNodes; end

        % Start off with what we have now
        pos = getNodePositions();

        % Get initial postions
        init_pos = getInitialPos(ui.status.currentFrame);

        % Set positions for specified nodes
        pos(node_idx,:) = init_pos(node_idx);

        % Update
        updateSkeleton(pos);
    end

%% Frame update and saving
    function markAllCorrect()
    % Helper for setting all nodes in the current frame as correct
        ui.status.movedNodes(:,:) = true;
        commitChanges();
    end
    function commitChanges()
    % Utility function for committing changes to node positions in the
    % current frame to the labels structure (but does not save to disk)
          
        % Get current positions
        pos = getNodePositions();

        % Check current status
        status = getStatus(ui.status.currentFrame);
        isLabeled = all(status == 2);
        
        for cam_ind=1:n_cams
            for i = horz(find(ui.status.movedNodes(:,cam_ind)))
                % Commit to labels
                labels.positions(i,:,cam_ind,ui.status.currentFrame) = pos(i,:,cam_ind);

                % Reset moved state
                ui.status.movedNodes(i,cam_ind) = false;

                % Mark unsaved changes
                ui.status.unsavedChanges(ui.status.currentFrame) = true;
            end
        end

        % Update status
        status = getStatus(ui.status.currentFrame);

        % Update skeleton display
        %updateSkeleton();

        % Update stats if changed
        if ~all(isLabeled) && all(status == 2)
            addToHistory(sprintf('Labeled frame %d', ui.status.currentFrame));
            numLabeled = numLabeled + 1;
            ui.status.framesLabeled.String = sprintf('Labeled: %d/%d (%.2f%%)', numLabeled, numFrames, numLabeled/numFrames*100);
        end

        % Update full status data
        ui.status.fullImg.CData(:,ui.status.currentFrame) = status;

        % Update zoomed status bar data
        zoomBoxIdx = ui.status.zoomImg.XData;
        if any(zoomBoxIdx == ui.status.currentFrame)
            ui.status.zoomImg.CData(:,zoomBoxIdx == ui.status.currentFrame) = status;
        end

        % Update status fig title
        savedStatus = '';
        if any(ui.status.unsavedChanges); savedStatus = ' [unsaved]'; end
        ui.status.fig.Name = sprintf('Status: %d/%d (%.2f%%) labeled%s', numLabeled, numFrames, numLabeled/numFrames*100, savedStatus);
    end
    function goToFrame(idx)
    % Utility function for seeking to another frame

        % Commit changes to labels
        commitChanges();
        
        % Autosave before anything
        if config.autoSave && ~isempty(ui.status.currentFrame) && ui.status.unsavedChanges(ui.status.currentFrame)
            saveLabels();
        end

        % Update image
        channels_pc=numChannels/n_cams;
        for camInd=1:n_cams
            % tsevi - change visualization
            im_2_show=box(:,:,(channels_pc*(camInd-1)+1):(channels_pc*(camInd-1)+3),idx);
            im_2_show(:,:,[1,3])=0.5*im_2_show(:,:,[1,3]);
            
            % show only 1 channel
            img_1_time = box(:,:,channels_pc*(camInd-1)+2,1);  

            ui.img(camInd).img.CData = im_2_show;

            % Update status
            ui.status.currentFrame = idx;
            ui.img(camInd).fig.Name = sprintf([num2str(camInd),'-%d/%d'], ui.status.currentFrame, numFrames);
        end

        % Get initial positions
        ui.status.initialPos = getInitialPos(idx);

        % Update with initial positions
        updateSkeleton(ui.status.initialPos);

        % Update status zoom box position
        zoomBoxIdx = zoomBoxWindow + ui.status.currentFrame;
        zoomBoxPts = [
            zoomBoxIdx(1) 0
            zoomBoxIdx(end) 0
            zoomBoxIdx(end) numNodes
            zoomBoxIdx(1) numNodes
            zoomBoxIdx(1) 0
            ];
        ui.status.fullZoomBox.XData = zoomBoxPts(:,1);
        ui.status.fullZoomBox.YData = zoomBoxPts(:,2);

        % Update zoomed status bar data
        ui.status.zoomImg.XData = zoomBoxIdx;
        ui.status.zoomImg.CData(:) = 0; % reset
        isValidIdx = zoomBoxIdx > 0 & zoomBoxIdx <= numFrames;
        ui.status.zoomImg.CData(:,isValidIdx) = ui.status.fullImg.CData(:,zoomBoxIdx(isValidIdx));
        ui.status.zoomAx.XLim = zoomBoxIdx([1 end]) + [-0.5 0.5];
    end

    function saveLabels()
    % Saves everything in the labels structure to disk

        stic;
        % Commit unsaved changes to labels
        commitChanges();

        % Update if there were any changes
        if any(ui.status.unsavedChanges)

            % Update last modified timestamp
            labels.lastModified = datestr(now);
        end

        % Save current frame so we pick up where we left off
        config.initialFrame = ui.status.currentFrame;

        % Save figure positions
        config.imgFigPos = ui.img(1).fig.Position;
        config.ctrlFigPos = ui.ctrl.fig.Position;
        config.statusFigPos = ui.status.fig.Position;


        % Save config
        labels.config = config;

        % Save to labels file
        save(labels.savePath, '-struct', 'labels')

        % Clear modified flags
        ui.status.unsavedChanges(:) = false;
        commitChanges();

        stocf('Saved labels: %s', labels.savePath)
    end

    function EpipolarIntersect()
        if sum(ui.status.movedNodes(ui.status.selectedNode,:))==2
            moved_cams=find(ui.status.movedNodes(ui.status.selectedNode,:));
            cams_2_intersect=find(~ui.status.movedNodes(ui.status.selectedNode,:));
            for cam_2_intersect=cams_2_intersect
                [~,line_inds(1)]=ismember([moved_cams(1),cam_2_intersect],coupless,'rows');
                [~,line_inds(2)]=ismember([moved_cams(2),cam_2_intersect],coupless,'rows');
                f = @(x) ui.epiLines(line_inds(1)).Function(x)-ui.epiLines(line_inds(2)).Function(x);
                x_intersect=fzero(f,frame_size/2);
                y_intersect=ui.epiLines(line_inds(1)).Function(x_intersect);
                
                % Update node position
                pos = getNodePositions();
                pos(ui.status.selectedNode,:,cam_2_intersect) = [x_intersect,y_intersect];
                updateSkeleton(pos,cam_2_intersect);
            end
        end
    end

%% Start!
goToFrame(config.initialFrame);
selectNode(1);
updateSkeleton();

end
