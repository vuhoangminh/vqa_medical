import json

file_train_path = "/mnt/Data/github/vqa_idrid/data/vqa/raw/annotations/mscoco_train2014_annotations.json"
file_val_path = "/mnt/Data/github/vqa_idrid/data/vqa/raw/annotations/mscoco_val2014_annotations.json"

with open(file_train_path) as f:
    content = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
# print(data)

# get the unique values
values_number = set()
values_other = set()
for item_content in content["annotations"]:
    if item_content['answer_type'] == "other":
        for item in item_content["answers"]:
            values_other.add(item['answer'])
    elif item_content['answer_type'] == "number":
        for item in item_content["answers"]:
            values_number.add(item['answer'])

# # the above uses less memory compared to this
# # since this has to create another array with all values
# # and then filter it for unique values
# values = set([i['score'] for i in content])

# its faster to save the results to a file rather than print them
with open('results.json', 'wb') as fid:
    # json cant serialize sets hence conversion to list
    json.dump(list(values_other), fid)
