import openpyxl
import math
import random


TRAIN_NAME = "training images"
VALIDATION_NAME = "validation images"
TESTING_NAME="testing images"

def split(targets,test_percentage,validation_percentage):
    test_num=int(math.floor(26*test_percentage))
    train_num=26-test_num
    validation_num=int(math.floor(train_num*validation_percentage))
    train_and_val_targets=set(random.sample(targets,train_num))
    test_targets=targets.difference(train_and_val_targets)
    validation_targets=set(random.sample(train_and_val_targets,validation_num))
    train_targets=train_and_val_targets.difference(validation_targets)
    return train_targets,validation_targets,test_targets

def write_to_file(setName,set_to_write,d):
    file = open(setName+".txt", "w")
    for targ in set_to_write:
        file.write("\n".join(d[targ]))
        file.close()

#test_percentage-Percentage of test set from all images
#validation_percentage-Percentage of validation set from all training images
def performSplit(test_percentage,validation_percentage):
    d = {}
    wb = openpyxl.load_workbook('driver_imgs_list.xlsx')
    sheet = wb.get_sheet_by_name('driver_imgs_list')
    for cellObj in sheet.columns[0]:
        target,category,image=cellObj.value.split(',')
        target=target.encode('ascii', 'ignore')
        category = category.encode('ascii', 'ignore')
        image = image.encode('ascii', 'ignore')
        value=category+","+image
        try:
            d[target].append(value)
        except KeyError:
            d[target] = [value]
    targets=set(d.keys())
    train_set,valid_set,test_set=split(targets,test_percentage,validation_percentage)
    write_to_file(TRAIN_NAME,train_set,d)
    write_to_file(VALIDATION_NAME,valid_set,d)
    write_to_file(TESTING_NAME,test_set,d)

