import xml.etree.ElementTree as ET


def get_data_xml(mydoc, tag):
    items = mydoc.getElementsByTagName(tag)
    value = items[0].firstChild.data
    return value


def get_object_xml(mydoc):
    items = mydoc.getElementsByTagName("object")
    boxes = list()
    for i in range(len(items)):
        item = items[i]
        boxes.append(get_bndbox_xml(item))
    return boxes


def get_bndbox_xml(item):
    object_name = get_data_xml(item, "name").lower()
    object_box = item.getElementsByTagName("bndbox")[0]
    xmin = int(get_data_xml(object_box, "xmin"))
    ymin = int(get_data_xml(object_box, "ymin"))
    xmax = int(get_data_xml(object_box, "xmax"))
    ymax = int(get_data_xml(object_box, "ymax"))
    bndbox_list = [object_name, xmin, xmax, ymin, ymax]
    return bndbox_list


def read_xml_breast(path):

    tree = ET.parse(path)

    root = tree.getroot()
    regions = root[0][1].findall('Region')

    pixel_spacing = float(root.get('MicronsPerPixel'))

    labels = []
    coords = []
    length = []
    area = []

    for r in regions:
        area += [float(r.get('AreaMicrons'))]
        length += [float(r.get('LengthMicrons'))]
        try:
            label = r[0][0].get('Value')
        except:
            label = r.get('Text')
        if 'benign' in label.lower():
            label = 1
        elif 'in situ' in label.lower():
            label = 2
        elif 'invasive' in label.lower():
            label = 3

        labels += [label]
        vertices = r[1]
        coord = []
        for v in vertices:
            x = int(v.get('X'))
            y = int(v.get('Y'))
            coord += [[x, y]]

        coords += [coord]

    return coords, labels, length, area, pixel_spacing
