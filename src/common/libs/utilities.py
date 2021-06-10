import os
import json
import statistics
import shutil
import subprocess
import hashlib

def checksum(fpath, htype='sha1'):
    if htype == 'sha1':
        h = hashlib.sha1()
    elif htype == 'sha256':
        h = hashlib.sha256()
    else:
        raise NotImplementedError(type)
    with open(fpath, 'rb') as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def cp(inpath, outpath):
    try:
        subprocess.run(('cp', '--reflink=auto', inpath, outpath))
    except FileNotFoundError:
        shutil.copy2(inpath, outpath)

def jsonfpath_load(fpath, default_type=dict, default=None):
    if not os.path.isfile(fpath):
        print('jsonfpath_load: warning: {} does not exist, returning default'.format(fpath))
        if default is None:
            return default_type()
        else:
            return default
    def jsonKeys2int(x):
        if isinstance(x, dict):
            return {k if not k.isdigit() else int(k):v for k,v in x.items()}
        return x
    with open(fpath, 'r') as f:
        return json.load(f, object_hook=jsonKeys2int)

def dict_to_json(adict, fpath):
    with open(fpath, "w") as f:
        json.dump(adict, f, indent=2)
        
def get_leaf(path: str) -> str:
    """Returns the leaf of a path, whether it's a file or directory followed by
    / or not."""
    return os.path.basename(os.path.relpath(path))

def get_root(fpath: str) -> str:
    '''
    return root directory a file (fpath) is located in.
    '''
    while fpath.endswith(os.pathsep):
        fpath = fpath[:-1]
    return os.path.dirname(fpath)

def avg_listofdicts(listofdicts):
    res = dict()
    for akey in listofdicts[0].keys():
        res[akey] = list()
    for adict in listofdicts:
        for akey, aval in adict.items():
            res[akey].append(aval)
    for akey in res.keys():
        res[akey] = statistics.mean(res[akey])

def list_of_tuples_to_csv(listoftuples, heading, fpath):
    with open(fpath, 'w') as fp:
        csvwriter = csv.writer(fp)
        csvwriter.writerow(heading)
        for arow in listoftuples:
            csvwriter.writerow(arow)

def filesize(fpath):
    return os.stat(fpath).st_size