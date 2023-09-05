from pysondb import getDb

todo_db = getDb('questions.json')

def addItem(new_item):
    item_id = todo_db.add(new_item)
    return item_id

data = todo_db.getAll()
print(data)