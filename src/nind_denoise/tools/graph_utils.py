def gen_markers(components):
    markers = []
    i = 0
    while len(markers) < len(components):
        markers.append("$%i$" % i)
        i+=1
    return markers

def make_markers_dict(components):
    markers = gen_markers(components)
    markersdict = dict()
    i = 0
    for acomp in components:
        markersdict[acomp] = markers[i]
        i += 1
        if i >= len(markers):
            i = 0
    return markersdict

def parse_log_file(path, smoothing_factor = 1, pre=None, post=None):
    data = []
    i = 0
    t = 0
    with open(path, 'r') as f:
        for l in f.readlines():
            added_data = False
            if pre is not None:
                if pre in l and post in l:
                    t += float(l.split(pre)[1].split(post)[0])
                    added_data = True
                elif pre in l:
                    t += float(l.split(pre)[1])
                    added_data = True
            elif 'Epoch' in l and 'nan' not in l:
                t += float(l.split(':')[-1])
                added_data = True
            elif 'loss = ' in l and 'time = ' not in l:
                t += float(l.split('loss = ')[-1])
                added_data = True
            if added_data:
                i += 1
                if i >= smoothing_factor:
                    data.append(t/smoothing_factor)
                    i = 0
                    t = 0
    print("Added %u points from %s"%(len(data), path))
    return data

