def get_kg2():
    # more simpler names
    kg_dict = {}
    # anatomical part
    kg_dict['right lung'] = 'Lung'
    kg_dict['right upper lung zone'] = 'Lung'
    kg_dict['right mid lung zone'] = 'Lung'
    kg_dict['right lower lung zone'] = 'Lung'
    kg_dict['right hilar structures'] = 'Lung'
    kg_dict['right apical zone'] = 'Lung'
    kg_dict['right costophrenic angle'] = 'Pleural'
    kg_dict['right hemidiaphragm'] = 'Pleural' # probably
    kg_dict['left lung'] = 'Lung'
    kg_dict['left upper lung zone'] = 'Lung'
    kg_dict['left mid lung zone'] = 'Lung'
    kg_dict['left lower lung zone'] = 'Lung'
    kg_dict['left hilar structures'] = 'Lung'
    kg_dict['left apical zone'] = 'Lung'
    kg_dict['left costophrenic angle'] = 'Pleural'
    kg_dict['left hemidiaphragm'] = 'Pleural' # probably

    kg_dict['trachea'] = 'Lung'
    kg_dict['right clavicle'] = 'Bone'
    kg_dict['left clavicle'] = 'Bone'
    kg_dict['aortic arch'] = 'Heart'
    kg_dict['upper mediastinum'] = 'Mediastinum'
    kg_dict['svc'] = 'Heart'
    kg_dict['cardiac silhouette'] = 'Heart'
    kg_dict['cavoatrial junction'] = 'Heart'
    kg_dict['right atrium'] = 'Heart'
    kg_dict['carina'] = 'Lung'

    return kg_dict