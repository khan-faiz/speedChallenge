import json
from decimal import *

def gen_gradient:
  pass

def gen_json:
  json_data = []
  with open("./data/test.txt") as txt_data:

      frame_i = 0
      for line in txt_data:
        data_row =  [ str(Decimal(frame_i) / Decimal(20)), line.replace('\n','') ]
        print(data_row)

        json_data.append(data_row)
        frame_i += 1

      
  with open("./data/test.json","w") as outfile:
      json.dump(json_data,outfile)
