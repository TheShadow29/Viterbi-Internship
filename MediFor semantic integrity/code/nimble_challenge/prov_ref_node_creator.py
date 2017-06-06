import pandas as pd
# import pdb


class prov_ref_node:
    def __init__(self, pfid):
        self.fid = pfid
        self.wfids = []
        return

    def append_wfids(self, wfid):
        self.wfids.append(wfid)
        return


class prov_nodes:
    def __init__(self, file_name):
        self.file_name = file_name
        self.nodes = []
        return

    def populate_data(self):
        f = open(self.file_name, 'rb')
        reader = pd.read_csv(f, sep='|')
        # print (reader.columns)
        # pdb.set_trace()
        # reader.ProvenanceProbeFileID
        prev_id = ''
        curr_id = ''
        for ind, row in reader.iterrows():
            # print(row['ProvenanceProbeFileID'])
            prev_id = curr_id
            curr_id = row['ProvenanceProbeFileID']
            if curr_id != prev_id and prev_id == '':
                curr_prov_ref_node = prov_ref_node(curr_id)
            elif curr_id != prev_id and prev_id != '':
                self.nodes.append(curr_prov_ref_node)
                curr_prov_ref_node = prov_ref_node(curr_id)
            curr_prov_ref_node.append_wfids(row['WorldFileID'])
        self.nodes.append(curr_prov_ref_node)
        f.close()


pref_file = '/arka_data/NC2017_Dev1_Beta4/reference/provenance/NC2017_Dev1-provenance-ref-node.csv'
# pnodes = prov_nodes(pref_file)
# pnodes.populate_data()
