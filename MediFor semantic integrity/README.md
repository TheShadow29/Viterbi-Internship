Code Structure for MediFor Project
==================================
cnn_caffe_places365/
--------------------
* places365_main.py : directly copied from the main evaluation code.
* places_MP.py : runs with multiprocessing library with speedup around 6x.

nimble_challenge/
-----------------
* compare_* : does a brute force comparison of feature vectors from the corresponding files
* create_prov_cluster : applies k-means for clustering
* create_prov_graph : creates graph with nodes as probe and world images, and edges with weight as the correlation between the two nodes
* data_copier : simple python script for copying images from one place to another
* disp_img : wrapper functions for using matplotlib imshow easily
* gen_manip* : automatically generates probe images randomly pasting an image onto another without smoothing effects
* get_all_features_* : to scrape all the features from the corresponding folder
* get_cropped_imgs : saves the cropped images from the Modified images in protest dataset
* get_recurrent_baseline : recurrently procures all the modified images and finally tries to get the base image
* hist_eq_and_testing.ipynb : ipython notebook for experimenting wheather histogram equalization gave any superior results
* imshow_coll : simple code to display images given in a list
* nimble_parser.ipynb : to parse csv files from the nimble dataset 
* node_contraction : experiment with contracting nodes
* parse_all_* : parsing csv files into pkl format
* plot_roc_cmc : plots roc and cmc curves
* probe_baseline_compare_* : comparing fv of probe and baseline
* prov_ref_node_creator : info storer for provenance csv files
* two_imgs_eff : compares all the metrics between two images

protest_dataset/
----------------
* ctpn/ : contains code for text detection
* check_iou : checks the iou between predicted boxes and gt
* egbis_wrapper : code to execute egbis binary file on the required images
* fcn_testing.ipynb : notebook to experiment with fcn segmentations
* invariace_testing.ipynb : naive attempt at checking the scale and contrast invariance of the cnn
* text_box_ctpn : code to generate text boxes using ctpn
* two_imgs_matcher : functions for comparing two images
* two_img_text_matcher : brute force search for all text match
* txt_data_storer : stores the information about txt bounding boxes for all files into a pkl file
