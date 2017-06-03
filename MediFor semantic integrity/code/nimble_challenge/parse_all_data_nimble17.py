# import caffe
import csv
# class nimble_index:
#     #Assumes manipulation by default
#     # def __init__(self,_task_id = 'manipulation',_probe_file_id = '',_probe_w = 1,_probe_h = 1):
#     def __init__(self, index_data):
#         # print (index_data)
#         self.task_id = index_data[0]
#         self.probe_file_id = index_data[1]
#         self.probe_file_name = index_data[2]
#         self.probe_w = index_data[3]
#         self.probe_h = index_data[4]
        
# class nimble_indices:
#     #Expects a csv file
#     def __init__(self,_file_name):
#         self.file_name = _file_name
#     def populate_data(self):
#         g = open(self.file_name)
#         csv_reader = csv.reader(g,delimiter=',')
#         self.field_names = csv_reader.next()
#         self.data = []
#         for dat in csv_reader:
#             ni = nimble_index(dat)
#             self.data.append(ni)
#         g.close()
class nimble_splice_reference:
    def __init__(self, reference_data):
        ###['TaskID', 'ProbeFileID', 'ProbeFileName', 'ProbeMaskFileName', 'BinaryProbeMaskFileName', 'ProbeBrowserFileName', 'DonorFileID', 'DonorFileName', 'DonorMaskFileName', 'DonorBrowserFileName', 'BaseFileName', 'BaseBrowserFileName', 'JournalName', 'IsTarget', 'ProjectDescription', 'ProjectType', 'PostprocessCompression', 'SemanticRepurposing', 'People', 'AntiforensicAddCamFingerprintPRNU', 'AudioClone', 'CompositePixelSize', 'AudioSplice', 'SeamCarving', 'ImageCompressionTable', 'ImageCompression', 'AudioActivity', 'TemporalOther', 'AudioVoiceOver', 'PostprocessCropFrames', 'AntiforensicAberrationCorrection', 'DataEmbeddingWatermark', 'AntiforensicCFACorrection', 'LaunderingSocialMedia', 'TemporalRemove', 'AntiforensicOther', 'AudioOthers', 'ImageReformat', 'AudioRemoval', 'AudioVoiceSwapping', 'SpatialOther', 'AntiforensicNoiseRestoration', 'TemporalReorder', 'SemanticRefabrication', 'SpatialSplice', 'SpatialClone', 'SpatialMovingObject', 'SpatialRemove', 'AntiforensicIllumination', 'LaunderingMedianFiltering', 'TemporalClone', 'Recapture', 'Mosaicing', 'ManipulationCategory', 'Natural', 'PostprocessStabilization', 'DataEmbeddingSteganography', 'SemanticRestaging', 'SpatialMovingCamera', 'TemporalSplice', 'FaceManipulations', 'ReflectionManipulations', 'ShadowManipulations']
        self.task_id = reference_data[0]
        self.probe_file_id = reference_data[1]
        self.probe_file_name = reference_data[2]
        self.probe_mask_file_name = reference_data[3]
        self.binary_probe_mask_file_name = reference_data[4]
        self.probe_browser_file_name = reference_data[5]
        self.donor_file_id = reference_data[6]
        self.donor_file_name = reference_data[7]
        self.donor_mask_file_name = reference_data[8]
        self.donor_browser_file_name = reference_data[9]
        self.base_file_name = reference_data[10]
        self.base_browser_file_name = reference_data[11]
        self.journal_name = reference_data[12]
        self.is_target = reference_data[13]
        self.project_description = reference_data[14]
        self.project_type = reference_data[15]
        self.postprocess_compression = reference_data[16]
        self.semantic_repurposing = reference_data[17]
        self.people = reference_data[18]
        self.antiforensic_add_cam_fingerprint_prnu = reference_data[19]
        self.audio_clone = reference_data[20]
        self.composite_pixel_size = reference_data[21]
        self.audio_splice = reference_data[22]
        self.seam_carving = reference_data[23]
        self.image_compression_table = reference_data[24]
        self.image_compression = reference_data[25]
        self.audio_activity = reference_data[26]
        self.temporal_other = reference_data[27]
        self.audio_voice_over = reference_data[28]
        self.postprocess_crop_frames = reference_data[29]
        self.antiforensic_aberration_correction = reference_data[30]
        self.data_embedding_watermark = reference_data[31]
        self.antiforensic_cfa_correction = reference_data[32]
        self.laundering_social_media = reference_data[33]
        self.temporal_remove = reference_data[34]
        self.antiforensic_other = reference_data[35]
        self.audio_others = reference_data[36]
        self.image_reformat = reference_data[37]
        self.audio_removal = reference_data[38]
        self.audio_voice_swapping = reference_data[39]
        self.spatial_other = reference_data[40]
        self.antiforensic_noise_restoration = reference_data[41]
        self.temporal_reorder = reference_data[42]
        self.semantic_refabrication = reference_data[43]
        self.spatial_splice = reference_data[44]
        self.spatial_clone = reference_data[45]
        self.spatial_moving_object = reference_data[46]
        self.spatial_remove = reference_data[47]
        self.antiforensic_illumination = reference_data[48]
        self.laundering_median_filtering = reference_data[49]
        self.temporal_clone = reference_data[50]
        self.recapture = reference_data[51]
        self.mosaicing = reference_data[52]
        self.manipulation_category = reference_data[53]
        self.natural = reference_data[54]
        self.postprocess_stabilization = reference_data[55]
        self.data_embedding_steganography = reference_data[56]
        self.semantic_restaging = reference_data[57]
        self.spatial_moving_camera = reference_data[58]
        self.temporal_splice = reference_data[59]
        self.face_manipulations = reference_data[60]
        self.reflection_manipulations = reference_data[61]
        self.shadow_manipulations = reference_data[62]
class nimble_man_reference:
    def __init__(self, reference_data):
        ###['TaskID', 'ProbeFileID', 'ProbeFileName', 'IsTarget', 'ProbeMaskFileName', 'ProbeBrowserFileName', 'BaseFileName', 'BaseBrowserFileName', 'JournalName', 'ProjectDescription', 'ProjectType', 'PostprocessCompression', 'SemanticRepurposing', 'People', 'AntiforensicAddCamFingerprintPRNU', 'AudioClone', 'CompositePixelSize', 'AudioSplice', 'SeamCarving', 'ImageCompressionTable', 'ImageCompression', 'AudioActivity', 'TemporalOther', 'AudioVoiceOver', 'PostprocessCropFrames', 'AntiforensicAberrationCorrection', 'DataEmbeddingWatermark', 'AntiforensicCFACorrection', 'LaunderingSocialMedia', 'TemporalRemove', 'AntiforensicOther', 'AudioOthers', 'ImageReformat', 'AudioRemoval', 'AudioVoiceSwapping', 'SpatialOther', 'AntiforensicNoiseRestoration', 'TemporalReorder', 'SemanticRefabrication', 'SpatialSplice', 'SpatialClone', 'SpatialMovingObject', 'SpatialRemove', 'AntiforensicIllumination', 'LaunderingMedianFiltering', 'TemporalClone', 'Recapture', 'Mosaicing', 'ManipulationCategory', 'Natural', 'PostprocessStabilization', 'DataEmbeddingSteganography', 'SemanticRestaging', 'SpatialMovingCamera', 'TemporalSplice', 'FaceManipulations', 'ReflectionManipulations', 'ShadowManipulations']
        self.task_id = reference_data[0]
        self.probe_file_id = reference_data[1]
        self.probe_file_name = reference_data[2]
        self.is_target = reference_data[3]
        self.probe_mask_file_name = reference_data[4]
        self.probe_browser_file_name = reference_data[5]
        self.base_file_name = reference_data[6]
        self.base_browser_file_name = reference_data[7]
        self.journal_name = reference_data[8]
        self.project_description = reference_data[9]
        self.project_type = reference_data[10]
        self.postprocess_compression = reference_data[11]
        self.semantic_repurposing = reference_data[12]
        self.people = reference_data[13]
        self.antiforensic_add_cam_fingerprint_prnu = reference_data[14]
        self.audio_clone = reference_data[15]
        self.composite_pixel_size = reference_data[16]
        self.audio_splice = reference_data[17]
        self.seam_carving = reference_data[18]
        self.image_compression_table = reference_data[19]
        self.image_compression = reference_data[20]
        self.audio_activity = reference_data[21]
        self.temporal_other = reference_data[22]
        self.audio_voice_over = reference_data[23]
        self.postprocess_crop_frames = reference_data[24]
        self.antiforensic_aberration_correction = reference_data[25]
        self.data_embedding_watermark = reference_data[26]
        self.antiforensic_cfa_correction = reference_data[27]
        self.laundering_social_media = reference_data[28]
        self.temporal_remove = reference_data[29]
        self.antiforensic_other = reference_data[30]
        self.audio_others = reference_data[31]
        self.image_reformat = reference_data[32]
        self.audio_removal = reference_data[33]
        self.audio_voice_swapping = reference_data[34]
        self.spatial_other = reference_data[35]
        self.antiforensic_noise_restoration = reference_data[36]
        self.temporal_reorder = reference_data[37]
        self.semantic_refabrication = reference_data[38]
        self.spatial_splice = reference_data[39]
        self.spatial_clone = reference_data[40]
        self.spatial_moving_object = reference_data[41]
        self.spatial_remove = reference_data[42]
        self.antiforensic_illumination = reference_data[43]
        self.laundering_median_filtering = reference_data[44]
        self.temporal_clone = reference_data[45]
        self.recapture = reference_data[46]
        self.mosaicing = reference_data[47]
        self.manipulation_category = reference_data[48]
        self.natural = reference_data[49]
        self.postprocess_stabilization = reference_data[50]
        self.data_embedding_steganography = reference_data[51]
        self.semantic_restaging = reference_data[52]
        self.spatial_moving_camera = reference_data[53]
        self.temporal_splice = reference_data[54]
        self.face_manipulations = reference_data[55]
        self.reflection_manipulations = reference_data[56]
        self.shadow_manipulations = reference_data[57]
class nimble_references:
    def __init__(self,_file_name):
        self.file_name = _file_name
        
    def populate_data(self):
        g = open(self.file_name)
        csv_reader = csv.reader(g,delimiter='|')
        self.field_names = csv_reader.next()
        self.data = []
        if "manip" in self.file_name:
            for dat in csv_reader:
                ni = nimble_man_reference(dat)
                self.data.append(ni)
        elif "splice" in self.file_name:
            for dat in csv_reader:
                ni = nimble_splice_reference(dat)
                self.data.append(ni)
        g.close()


# def parse_index_file(file_type):
    # if file_type == 'manipulation':
    #     # f = open('/arka_data/NC2016_Test0613/indexes/NC2016-manipulation-index_new.csv')
    # elif file_type == 'removal':
    #     # f = open('/arka_data/NC2016_Test0613/indexes/NC2016-removal-index_new.csv')
    # elif file_type == 'splice':
    #     # f = open('/arka_data/NC2016_Test0613/indexes/NC2016-splice-index_new.csv')
    # else:
    #     return -1
    #Assuming it is manipulation file for now
    # field_names = f.readline()
    # data = f.readlines()

# if __name__ == '__main__':
    # parse_index_file('manipulation')
# man_index_file = '/arka_data/NC2016_Test0613/indexes/NC2016-manipulation-index_new.csv'
# rem_index_file = '/arka_data/NC2016_Test0613/indexes/NC2016-removal-index_new.csv'
# splice_index_file = '/arka_data/NC2016_Test0613/indexes/NC2016-splice-index_new.csv'
# man_nimble_indices = nimble_indices(man_index_file)
# man_nimble_indices.populate_data()
# rem_nimble_indices = nimble_indices(rem_index_file)
# rem_nimble_indices.populate_data()
# splice_nimble_indices = nimble_indices(splice_index_file)
# splice_nimble_indices.populate_data()

# man_ref_file = '/arka_data/NC2016_Test0613/reference/manipulation/NC2016-manipulation-ref_new.csv'
man_ref_file = '/arka_data/NC2017_Dev1_Beta4/reference/manipulation-image/NC2017_Dev1-manipulation-image-ref.csv'
# rem_ref_file = '/arka_data/NC2016_Test0613/reference/removal/NC2016-removal-ref_new.csv'
# splice_ref_file = '/arka_data/NC2016_Test0613/reference/splice/NC2016-splice-ref_new.csv'
splice_ref_file = '/arka_data/NC2017_Dev1_Beta4/reference/splice/NC2017_Dev1-splice-ref.csv'
# man_nimble_ref = nimble_references(man_ref_file)
# man_nimble_ref.populate_data()
# rem_nimble_ref = nimble_references(rem_ref_file)
# rem_nimble_ref.populate_data()
# splice_nimble_ref = nimble_references(splice_ref_file)
# splice_nimble_ref.populate_data()
    
