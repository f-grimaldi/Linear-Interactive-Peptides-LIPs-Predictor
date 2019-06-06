# Import required modules
import logging
# Import default libraries
import pandas as pd
import numpy as np
import requests
import json
import zipfile
import time
import warnings
# Import BioPython utils
from Bio.PDB import PDBList, calc_angle, calc_dihedral, PPBuilder, is_aa, PDBIO, NeighborSearch, DSSP, HSExposureCB
from Bio.PDB.PDBParser import PDBParser
# Import other methods
from sklearn.feature_extraction.text import CountVectorizer
from scipy import signal

# Function for tagging residues as LIP/non-LIP
# Overwrites entries in the second dataset with LIP flag from the first dataset
def LIP_tag(ds_original, ds_residues):
    #For every protein we take the information of where LIP residue are
    for idx, row in ds_original.iterrows():
        # Bind information to correct variables
        pdb, chain, start, end = (row[0:4])
        # Skips if start and and are 'neg'
        if (start == 'neg') or (end == 'neg'):
            continue
        # Cast start and end values to integer
        start, end = int(start), int(end)
        # Get the correct slice of data which will be edited
        sliced = ((ds_residues['PDB_ID'] == pdb)
                    & (ds_residues['CHAIN_ID'] == chain)
                    & (ds_residues['RES_ID'] <= end)
                    & (ds_residues['RES_ID'] >= start))
        #Now we set to 1 all the residue whose features are the one desired
        ds_residues.loc[sliced, 'LIP'] = 1
        ds_residues.loc[sliced, 'LIP_SCORE'] = 1


# Function for extracting DSSP data from a list of pdb_ids
def get_DSSP(pdb_ids, pdb_dir='./'):
    # Check parameters
    logging.debug('PDB ids:')
    logging.debug(pdb_ids)
    logging.debug('PDB directory: \'{}\''.format(pdb_dir))
    # Define a list of dssp features (which will be stored in a list before being turned into DataFrame
    dssp_list = list()
    # Loop thorugh every protein
    for pdb_id in pdb_ids:
        # Parse structure of the protein
        structure = PDBParser(QUIET=True).get_structure(pdb_id, pdb_dir + '/pdb{}.ent'.format(pdb_id))
        # Get only first model
        model = structure[0]
        # Define DSSP instance of the 0-th model
        dssp = DSSP(model, pdb_dir + '/pdb{}.ent'.format(pdb_id), dssp="/usr/local/bin/mkdssp")
        # Get DSSP features: dssp index, amino acid, secondary structure, relative ASA, phi, psi, NH_O_1_relidx,
        # NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy, NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy
        # Get chain id and residue id
        for ids, res in zip(dict(dssp), dssp):
            # Create the DSSP row
            row = list()
            row.append(pdb_id)
            row.extend(list(ids))
            row.extend(list(res))
            # Add row to dssp list
            dssp_list.append(row)

    # Define feature names
    columns = ['PDB_ID', 'CHAIN_ID', 'RES_ID', 'DSSP_ID', 'AA', 'SEC_STRUCT', 'REL_ASA', 'PHI', 'PSI',
               'NH_O_1_relidx', 'NH_O_1_energy', 'O_NH_1_relidx', 'O_NH_1_energy', 'NH_O_2_relidx',
               'H_O_2_energy', 'O_NH_2_relidx', 'O_NH_2_energy']

    # Define DSSP DataFrame
    ds_dssp = pd.DataFrame(dssp_list, columns=columns)
    # Turn RES_ID from tuple to integer (gets 1-st element)
    ds_dssp.RES_ID = ds_dssp.RES_ID.apply(lambda x: x[1])
    # Return DSSP dataset
    return ds_dssp

# Function for dwonloading RING archives features from one or more pdb files/ids
def download_RING(pdb_ids, download_dir='.', time_sleep=5):
    # Initialize the status of every RING server job into a dict pdb_id -> status
    job_status = ['nd' for pdb_id in pdb_ids]
    # Initialize job_ids returned from RING server
    job_ids = ['nd' for pdb_id in pdb_ids]
    # Loop through every protein
    for i in range(0, len(pdb_ids)):
        # Define POST request parameters to tell RING web service to start elaborating
        req = {"pdbName": pdb_ids[i], "chain": "all", "seqSeparation": "5", "networkPolicy": "closest",
               "nowater": "true", "ringmd": "false", "allEdges": "true",
               "thresholds": '{"hbond": 3.5, "vdw": 0.5, "ionic": 4, "pipi": 6.5, "pication": 5, "disulphide": 2.5}'}
        # Define a request object to handle connection
        r = requests.post('http://protein.bio.unipd.it/ringws/submit',
                          data=json.dumps(req),
                          headers={'content-type': 'application/json'})
        # Define id of the job provided by RING
        job_ids[i] = json.loads(r.text)['jobid']
    # Debug info
    logging.debug(zip(pdb_ids, job_ids))
    # Repeat until every job is complete
    while any([(js != 'complete') for js in job_status]):
        # Loop through every request to check its status
        for i in range(0, len(pdb_ids)):
            # Proceed only if not already completed
            if job_status[i] != 'complete':
                # Retrieve (update) current server elaboration status for this specific job
                job_status[i] = json.loads(
                    requests.get('http://protein.bio.unipd.it/ringws/status/{}'.format(job_ids[i])).text).get("status", None)
                # Skip if already downloaded
                if job_status[i] == 'complete':
                    # Define archive file name
                    archive_file = "{}_network.zip".format(pdb_ids[i])
                    # Download RING output as an archive
                    r = requests.get("http://protein.bio.unipd.it/ring_download/{}/{}".format(job_ids[i], archive_file))
                    # Write the archive file
                    with open(download_dir + '/' + archive_file, "wb") as fout:
                        fout.write(r.content)
            # Debug info
            logging.debug("Status for pdb {}: {} with job id {}".format(pdb_ids[i], job_status[i], job_ids[i]))
        # Wait for server elaboration
        time.sleep(time_sleep)


# Creates a DataFrame of RING features only
def get_RING(pdb_ids, pdb_dir='.', ring_dir='.', contact_threshold=3.5):
    # Define columns
    # Define a list of RING features
    ring_list = list()
    # Loop through every PDB ID passed as argument
    for pdb_id in pdb_ids:

        # Parse structure
        structure = PDBParser(QUIET=True).get_structure(pdb_id, pdb_dir + '/' + 'pdb{}.ent'.format(pdb_id))

        # Create chains with residue ids compatible with RING
        nodes = {}
        for chain in structure[0]:
            nodes[chain.id] = []
            for residue in chain:
                if residue.id[0] == ' ':
                    node = "{}:{}:{}:{}".format(chain.id,
                                                residue.id[1],
                                                residue.id[2] if residue.id[2] != ' ' else '_',
                                                residue.get_resname())
                    nodes[chain.id].append(node)

        # Opens ZIP file
        zf = zipfile.ZipFile(ring_dir + '/' + '{}_network.zip'.format(pdb_id), 'r')
        # Opens edges file
        with zf.open('{}_edges.txt'.format(pdb_id), 'r') as f:
            # Skip first row: header
            next(f)
            # Reads edges line by line
            for line in f:
                # Encodes string as utf8 from bytes
                line = line.decode('utf-8')
                # node = chain:residue_number:insertion_code:residue_name
                # edge = localization:contact_type (localization MC = main chain, SC = side chain)
                node_a, edge, node_b, distance, _, _, atom_a, atom_b = line.split(sep='\t')[0:8]
                distance = float(distance)
                if distance <= contact_threshold:
                    node_a = node_a.split(':')
                    node_b = node_b.split(':')
                    edge_type, edge_loc = edge.split(':')
                    # Create a contact instance: (PDB_ID, CHAIN_ID(A), RESIDUE_ID(A), CHAIN_ID(B), RESIDUE_ID(B), other RING features)
                    ring_list.append((pdb_id, node_a[0], node_a[1], node_b[0], node_b[1], edge_loc, atom_b, atom_a, edge_type))
                    ring_list.append((pdb_id, node_b[0], node_b[1], node_a[0], node_a[1], edge_loc, atom_a, atom_b, edge_type))

        # Define RING features columns
        columns = ['PDB_ID', 'CHAIN_ID_A', 'RES_ID_A', 'CHAIN_ID_B', 'RES_ID_B', 'EDGE_LOC', 'ATOM_A', 'ATOM_B', 'EDGE_TYPE']
        # Turn RING features list into DataFrame
        ds_ring = pd.DataFrame(ring_list, columns=columns)
        return ds_ring


def res2features(df):
    """
    REQUIRE:
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    INPUT:
    A dataframe containing a RES_NAME columns

    OUTPUT:
    The same dataframe merged with a matrix of the output of CountVectorize of RES_NAME
    """

    #Transform from obj to str
    res_name = df.RES_NAME.apply(lambda x: str(x))
    #Apply CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(res_name)
    #Create dataframe of CountVectorize
    df_res = pd.DataFrame(X.toarray(), columns=[i.upper() for i in vectorizer.get_feature_names()])
    #Merge
    df1 = pd.merge(df, df_res, left_index=True, right_index=True)

    return df1


def struct2features(df):
    """
    REQUIRE:
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    INPUT:
    A dataframe containing a SEC_STRUCT columns

    OUTPUT:
    The same dataframe merged with a matrix of the output of CountVectorize of SEC_STRUCT

    WARNING:
    CountVectorize remove words too short such our tags of the secondary structure, for this reason we need to
    substitute the tags with their original meaning.
    See: https://biopython.org/DIST/docs/api/Bio.PDB.DSSP%27-pysrc.html for the legend of the tags
    """

    df.loc[df.SEC_STRUCT == '-', 'SEC_STRUCT'] = "NO_STRUCT"
    df.loc[df.SEC_STRUCT == '0', 'SEC_STRUCT'] = "ZERO"
    df.loc[df.SEC_STRUCT == 'B', 'SEC_STRUCT'] = 'ISOLATED_BETA_BRIGE'
    df.loc[df.SEC_STRUCT == 'E', 'SEC_STRUCT'] = 'STRAND'
    df.loc[df.SEC_STRUCT == 'G', 'SEC_STRUCT'] = '3-10_ELIX'
    df.loc[df.SEC_STRUCT == 'H', 'SEC_STRUCT'] = 'ALPHA_ELIX'
    df.loc[df.SEC_STRUCT == 'I', 'SEC_STRUCT'] = 'PI_ELIX'
    df.loc[df.SEC_STRUCT == 'S', 'SEC_STRUCT'] = 'BEND'
    df.loc[df.SEC_STRUCT == 'T', 'SEC_STRUCT'] = 'TURN'
    sec_struct = df.SEC_STRUCT
    #Apply CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sec_struct)
    #Create dataframe of CountVectorize
    df_ss = pd.DataFrame(X.toarray(), columns=[i.upper() for i in vectorizer.get_feature_names()])
    #Merge
    df1 = pd.merge(df, df_ss, left_index=True, right_index=True)

    return df1


def contacts2features(df, ignore_warnings = True):
    """
    REQUIRE:
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    INPUT:
    df = dataframe of main features with columns EDGE_TYPE and EDGE_LOC

    OUTPUT:
    The main features dataframe with CountVectorize of EDGE_LOC and EDGE_TYPE
    """
    #Ignore warnings
    if ignore_warnings:
        warnings.filterwarnings("ignore")

    #NaN transformed to a categoricacl value
    df.EDGE_LOC[df.EDGE_LOC.isna()] = "NO_EDGE_LOC"
    df.EDGE_TYPE[df.EDGE_TYPE.isna()] = "NO_EDGE_TYPE"
    edge_loc = df.EDGE_LOC
    edge_type = df.EDGE_TYPE
    #Call object
    vectorizer = CountVectorizer()
    #One-Hot-Enc of EDGE_LOC
    X_loc = vectorizer.fit_transform(edge_loc)
    df_loc = pd.DataFrame(X_loc.toarray(),
                          columns=[i.upper() for i in vectorizer.get_feature_names()])
    #One-Hot-Enc of EDGE_TYPE
    X_type = vectorizer.fit_transform(edge_type)
    df_type = pd.DataFrame(X_type.toarray(),
                          columns=[i.upper() for i in vectorizer.get_feature_names()])
    #Merge LOC and TYPE
    df_rings = pd.merge(df_loc, df_type,
                       left_index=True,
                       right_index=True)
    #Merge to original dataframe
    df_final = pd.merge(df, df_rings, left_index=True, right_index=True)

    #Restore warnings
    warnings.filterwarnings('default')

    return df_final


def chain_len(df, get_time = False, ignore_warnings = False):
    """
    REQUIRE:
    import pandas as pd

    INPUT:
    df = dataframe of main features with CHAIN_ID and PDB_ID
    time = If true print the time needed to complete the procedure (default False)
    warnings = If true it will print the warnings (default False)

    OUTPUT:
    A dataframe with the variable CHAIN_LEN as feature
    """
    #Ignore warnings
    if ignore_warnings:
        warnings.filterwarnings("ignore")

    #Get matrix ready
    df1 = df
    df1['CHAIN_LEN'] = np.array(0)

    #Start time
    start = time.time()

    #Start cycle where for each chain we add the chains lenghts to all the resiudes of that chain
    for pdb_id in set(df.PDB_ID.unique()):
        for chain in set(df.CHAIN_ID[df.PDB_ID == pdb_id]):
            len_chain = len(df.CHAIN_ID[(df.CHAIN_ID == chain) & (df.PDB_ID == pdb_id)])
            df1['CHAIN_LEN'][(df.PDB_ID == pdb_id) & (df.CHAIN_ID == chain)] = len_chain

    #Print time if passed as arg
    if get_time:
        print('Time passed: {}'.format(time.time() - start))

    #Restore warnings
    warnings.filterwarnings('default')

    #Return
    return df1
