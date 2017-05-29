# import caffe
import csv
class nimble_index:
    #Assumes manipulation by default
    # def __init__(self,_task_id = 'manipulation',_probe_file_id = '',_probe_w = 1,_probe_h = 1):
    def __init__(self, index_data):
        # print (index_data)
        self.task_id = index_data[0]
        self.probe_file_id = index_data[1]
        self.probe_file_name = index_data[2]
        self.probe_w = index_data[3]
        self.probe_h = index_data[4]
        
class nimble_indices:
    #Expects a csv file
    def __init__(self,_file_name):
        self.file_name = _file_name
    def populate_data(self):
        g = open(self.file_name)
        csv_reader = csv.reader(g,delimiter=',')
        self.field_names = csv_reader.next()
        self.data = []
        for dat in csv_reader:
            ni = nimble_index(dat)
            self.data.append(ni)
        g.close()

class nimble_reference:
    def __init__(self, reference_data):
        #['TaskID', 'ProbeFileID', 'ProbeFileName', 'ProbeMaskFileName', 'DonorFileID', 'DonorFileName', 'DonorMaskFileName', 'IsTarget', 'ProbePostProcessed', 'DonorPostProcessed', 'ManipulationQuality', 'IsManipulationTypeRemoval', 'IsManipulationTypeSplice', 'IsManipulationTypeCopyClone', 'Collection', 'BaseFileName', 'Lighting', 'IsControl', 'CorrespondingControlFileName', 'SemanticConsistency']
        self.task_id = reference_data[0]
        self.probe_file_id = reference_data[1]
        self.probe_file_name = reference_data[2]
        self.probe_mask_file_name = reference_data[3]
        self.donor_file_id = reference_data[4]
        self.donor_file_name = reference_data[5]
        self.donor_mask_file_name = reference_data[6]
        self.is_target = reference_data[7]
        self.probe_post_processed = reference_data[8]
        self.donor_post_processed = reference_data[9]
        self.manipulation_quality = reference_data[10]
        self.is_manipulation_type_removal = reference_data[11]
        self.is_manipulation_type_splice = reference_data[12]
        self.is_manipulation_type_copy_clone = reference_data[13]
        self.collection = reference_data[14]
        self.base_file_name = reference_data[15]
        self.lighting = reference_data[16]
        self.is_control = reference_data[17]
        self.corresponding_control_file_name = reference_data[18]
        self.semantic_consistency = reference_data[19]
            
class nimble_references:
    def __init__(self,_file_name):
        self.file_name = _file_name
        
    def populate_data(self):
        g = open(self.file_name)
        csv_reader = csv.reader(g,delimiter=',')
        self.field_names = csv_reader.next()
        self.data = []
        for dat in csv_reader:
            ni = nimble_reference(dat)
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

if __name__ == '__main__':
    # parse_index_file('manipulation')
    man_index_file = '/arka_data/NC2016_Test0613/indexes/NC2016-manipulation-index_new.csv'
    rem_index_file = '/arka_data/NC2016_Test0613/indexes/NC2016-removal-index_new.csv'
    splice_index_file = '/arka_data/NC2016_Test0613/indexes/NC2016-splice-index_new.csv'
    man_nimble_indices = nimble_indices(man_index_file)
    man_nimble_indices.populate_data()

    man_ref_file = '/arka_data/NC2016_Test0613/reference/manipulation/NC2016-manipulation-ref_new.csv'
    rem_ref_file = '/arka_data/NC2016_Test0613/reference/removal/NC2016-removal-ref_new.csv'
    splice_ref_file = '/arka_data/NC2016_Test0613/reference/splice/NC2016-splice-ref_new.csv'

    man_nimble_ref = nimble_references(man_ref_file)
    man_nimble_ref.populate_data()
