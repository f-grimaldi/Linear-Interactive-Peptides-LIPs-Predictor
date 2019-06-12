# sbio_lip_predictor

Software for predicting LIP tagged aminoacidic residues

## Requirements

## Usage

LIP_predictor is composed of two main scripts:

1. train.py
2. predict.py

*Note*: command are given from inside LIP_predictor directory in this guide.

### Training

Basic training: takes as input the model saved as default in ./model_files/default.joblib and retrains it with given data.

```
python3 ./train.py [options] [lip_file]
```

1. lip_file: path to training file, which contains proteins, chains LIP and non-LIP tag instruction. String;
2. -e, --exclude: allows to exclude some proteins from the given lip_file. String;
3. -ws, --window_size: define the size of the window used to compute an average of the residues features. Int;
4. -rf, --ring_force: forces to download RING data, even if it has already been downloaded. Int in {0,1;
5. -rd, --ring_dir: define directory where RING data is stored. String;
6. -pd, --pdb_dir: define directory where PDB data is stored. String;
7. -cf, --config_file: define a configuration file from which other settings will be loaded. JSON;

### Prediction

Basic prediction takes as input a PDB id and computes LIP score and LIP flag for every aminoacidic residue in the given protein.

```
python3 ./predict.py [options] [pdb_id]
```

1. pdb_id: PDB id of the protein for which LIP tags will be predicted. String;
3. -ws, --window_size: define the size of the window used to compute an average of the residues features. Int;
4. -rf, --ring_force: forces to download RING data, even if it has already been downloaded. Int in {0,1;
5. -rd, --ring_dir: define directory where RING data is stored. String;
6. -pd, --pdb_dir: define directory where PDB data is stored. String;
7. -cf, --config_file: define a configuration file from which other settings will be loaded. JSON;

### Configuration file

It is possible to provide a custom configuration file for either train.py and predict., JSON formatted. Custom configuration file entries will be overwritten by command line parameters, which are considered to have higher priority.

Inside custom configuration file, it is possible to spcify which kind of model we want to be trained by defining 'model' entry as a dictionary, which has two parameters itself: 'name' and 'args'.
1. model.name: Name of the scikit-learn classifier which must be trained. String in ['RandomForestClassifier', 'LogisticRegression', 'MLPCLassifier', 'KNeighborsClassifier', 'SVC', 'QuadraticDiscriminantAnalysis'];
2. model.args: Arguments for scikit-learn classifier chosen. Dict;

## Features extraction

### General approach

LIP_ppredictor adopts a Relational Database kind of approach. First of all, we extract residues we are interested in from PDB database, then we extract features from DSSP application and RING service. We store either PDB information, RING and DSSP features into tables in which the triple (PDB_ID, CHAIN_ID, RES_ID) is the unique identifier of the row. Given this assumption, we build a complete dataset of features by joining every table on the unique identifier trible described above.

Since we work with tables, we strongly leveraged Pandas library provided by Anaconda3 and Python3.

### PDB

PDB database is downloaded from server using BioPython utils, saved in the appropriate directory (by default ./pdb_files) with extension .ent. Afterwards, it gets read from the stored file and loaded into a Pandas DataFrame.

Table:
PDB_ID: id of the protein; String;
MODEL_ID: model contained into the protein; Int;
CHAIN_ID: chain contained in the model. We kept only the 0-th model; Char;
CHAIN_LEN: length of the chain. Equal value for every entry given same triple (PDB_ID, MODEL_ID, CHAIN_ID); Int;
RES_ID: residue id of a specific aminoacid. String casted to Int;
RES_NAME: name of the aminoacid which composes the residue; String;
LIP_SCORE: probability of being a Linear Interacting Peptide, assigned by a trained ML model (in case of prediction) or manually flagged (in case of model training). Takes values in [0,1];
LIP: defines if a residue is a LIP. Takes values in {0,1};

### DSSP

DSSP features extracted using DSSP program through BioPython DSSP interface. Retrieved data are immediately inserted into a Pandas DataFrame.

Table:
PDB_ID: see PDB;
CHAIN_ID: see PDB;
RES_ID: see PDB;
SEC_STRUCT: defines to which secondary structure the residue belongs. String;
REL_ASA: RELative Absolute Solvent Accessibility measusres the surface of the residue exposed to the solvent. Takes values in [0,1];
PHI, PSI: dihedral angles of the aminoacid, expressed in degrees. If the angle does not exist, its value will be 360. Takes values in [0, 360]
NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy, NH_O_2_relidx, H_O_2_energy, O_NH_2_relidx, O_NH_2_energy: energy values. Double;

### RING

RING features are extracted by a request to RING APIs. First, the "elaborate" POST request is sent by our application acting as a client. If request responds with a 200 OK status, then client starts polling in order to intercept termination of server's computational phase. Afterwards, computed information is downloaded as a file with .zip extension. Downloaded file is then unzipped and read, and RING table is defined by means of Pandas DataFrame.

Table:
PDB_ID: see PDB;
CHAIN_ID: see PDB;
RES_ID: see PDB;
EDGE_LOC: locations where contacts are formed; Strings separated by space charachter ('');
EDGE_TYPE: type of contacts; Strings separated by space charachter (' ');
INTRA_CONTACTS: number of intra-chain contacts formed by the aminoacid. Int;
INTER_CONTACTS: number of inter-chain contacts formed by the aminoacid. Int;
INTRA_INTER_CONTACTS: intra-chain contacts / inter-chain contacts ratio of the aminoacid. Double;
