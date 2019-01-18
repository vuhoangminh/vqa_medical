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
