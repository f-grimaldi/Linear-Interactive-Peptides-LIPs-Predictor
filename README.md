# sbio_lip_predictor

Software for predicting LIP tagged aminoacidic residues

## Requirements

## Usage

LIP_predictor is composed of two main scripts:

    1. *train.py*
    2. *predict.py*

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

*./config.json*

It is possible to provide a custom configuration file for either train.py and predict., JSON formatted. Custom configuration file entries will be overwritten by command line parameters, which are considered to have higher priority.

Inside custom configuration file, it is possible to spcify which kind of model we want to be trained by defining 'model' entry as a dictionary, which has two parameters itself: 'name' and 'args'.

    1. model.name: Name of the scikit-learn classifier which must be trained. String in ['RandomForestClassifier', 'LogisticRegression', 'MLPCLassifier', 'KNeighborsClassifier', 'SVC', 'QuadraticDiscriminantAnalysis'];
    2. model.args: Arguments for scikit-learn classifier chosen. Dict;

## Features extraction

*./modules/feature_extraction.py*

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


## Feature pre-proccesing

*./modules/feature_preprocessing.py*

### Handling Categorical Features

Categorical Features such the name of the residues or the type of secondary structure, have been handled using One-Hot-Encoding procedure (https://en.wikipedia.org/wiki/One-hot). This procedure create one feature for each level of the categorical feature, assigning *1* to the features that represent the categorical values of our instances and *0* to all the others features.

This procedure has been done for the following features: <br>
      
    1. Type of residues
    2. Type of secondary structure
    3. Type of contacts
    
### Sliding Window

In order to get information of the context, a sliding windows by a rolling procedure (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html) with gaussian filtering has been applied to our instances (residues). <br> Every value has been substituted by a gaussian mean of windows *k* centered in that value. In this way we take in consideration also the values of the *k* closest residues, assigning to them a multiplicative factor which goes to 0 as the distance increases and is *1* for the centered value.

Parameters of sliding windows:

    1. windows size (integer, default = 5): size of the windows (usually an odd number for symmetric purpose). 
    2. std (float, default = 1): standard deviation of the gaussian filter. Higher values mean a minor decrease of the multiplicative factor as the distance increase.


N.B.
For the first values and the last values of every chain, it has been create a mirroring of their next/previous residues. E.G:
       
     res1, res2, res3 --> res3, res2, res1, res2, res3
     
## Model

*./modules/models*

### Leave One Out Cross Validation Protein Based (LOO-CV Portein Based)

In order to validated our model a *LOO-CV* has been applied. In our case the "one left out" wasn't a single instances but a whole protein. <br>
For protein *p* we trained the model with all the proteins available except the protein *p* and then we used as test that protein *p*.

### Feature Selection

A Random Forest Classifier has been used to extract the best features. Since the type of contacts extracted from the RING server didn't result to have meaningful impact (less than *0.001%*) we excluded them from the input matrix. 

### Models

Various algorithms has been used with various parameters (k-Nearest Neighbours, Support Vector Classifier, Linear Discriminant Analysis, Quadratic Discriminant Analysis, MultiLayerPerceptron, Random Forest, Decision Tree, AdaBoost, Logistic Regression). 

The best performing algorithm have been *Random Forest Classifier* and the *Multi-Layer Perceptron* (MLP) with a balanced accuray of over *0.90* and a f1 score greater than *0.85*.

In the end it has been decided to use a *Random Forest Classifier* because of better performance and less training time required. <br> 
After a grid search the best parameters apperead to be a *sliding windows between 3 and 7*, a standard deviation of the *gaussian filtering around 1*, a *number of estimator around 100* (80-120) and *no limits to trees depth and number of leaves*.

___FINAL MODEL___

    1. Sliding Windows:
       a) Windows Size = 5
       b) Standard Deviation = 1
    2. Classifier: 
       a) Type = Random Forest Classifier:
       b) Parameters = {n_estimator: 100}
  
___CURRENT VALIDATION RESULT___

For every protein has been computed balanced accuracy and f1 score and the final results are the avarage of this two scores of every protein. Random seed not set 

    1. Balanced Accuracy: 0.930 
    2. F1-Score: 0.894