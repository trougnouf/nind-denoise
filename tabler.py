# results collection helper
import json
import statistics


# ssim avg, min,

def tablemaker(datafn='res.json', blacklistimages=None, whitelistimages=None):
    with open(datafn) as f:
        data = json.load(f)
    ndata = dict()
    images=[]
    for image in data[list(data.keys())[0]]['results'].keys():
        if blacklistimages is None or (blacklistimages is not None and blacklistimages not in image):
            if whitelistimages is None or whitelistimages in image:
                images.append(image)
    for experiment in data.keys():
        ndata[experiment] = dict()
        for image in images:
            if image not in data[experiment]['results'].keys():
                #data.remove(experiment)
                continue
            for isoval in data[experiment]['results'][image].keys():
                nisoval = 'High ISO' if 'H' in isoval else isoval
                nisoval = nisoval.split('-')[0]
                if nisoval not in ndata[experiment].keys():
                    ndata[experiment][nisoval] = {'ssim': [], 'mse': []}
                for metric in ['ssim', 'mse']:
                    ndata[experiment][nisoval][metric].append(data[experiment]['results'][image][isoval][metric])
    isovals = []
    tbl = 'ISO value,'
    for isoval in ndata[list(ndata.keys())[0]].keys():
            tbl+=isoval+','
            isovals.append(isoval)
    tbl = tbl[:-1]+'\n'
    if len(images)>1:
        tbl+='# images,'
        for isoval in isovals:
                tbl+=str(len(ndata[list(ndata.keys())[0]][isoval]['ssim']))+','
        tbl = tbl[:-1]+'\n'
    for experiment in ndata.keys():
        tbl+=experiment+','
        for isoval in isovals:
            try:
                tbl+=str(statistics.mean(ndata[experiment][isoval]['ssim']))+','
            except KeyError:
                tbl+=','
        tbl = tbl[:-1]+'\n'
    print(tbl)
    print('\n')
tablemaker('res.json', blacklistimages='C500D')
tablemaker('res1.json', blacklistimages='C500D')
tablemaker('res1.json', whitelistimages='C500D')
tablemaker('res2.json')
