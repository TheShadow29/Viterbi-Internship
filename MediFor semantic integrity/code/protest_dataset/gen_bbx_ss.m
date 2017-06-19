function [boxes] = gen_bbx_ss(img)
    addpath('../../../SelectiveSearchCodeIJCV/Dependencies/')
    simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
    simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies

    colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
    colorType = colorTypes{1}; % Single color space for demo

    % Here you specify which similarity functions to use in merging
    simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
    simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies

    % Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
    % Note that by default, we set minSize = k, and sigma = 0.8.
    k = 200; % controls size of segments of initial segmentation. 
    minSize = k;
    sigma = 0.8;

    % As an example, use a single image
    % images = {'000015.jpg'};
%     im = imread(images{1});
    im = img;

    % Perform Selective Search
    [boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);
    % boxes = boxes(1,:)
    % Show boxes
    % ShowRectsWithinImage(boxes, 5, 5, im);
end