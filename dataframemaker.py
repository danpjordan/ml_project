import pandas as pd

DISCRETE_THRESHOLD = 5

class Column:
  def __init__(self, is_String, is_Discrete):
    self.is_String = is_String
    self.is_Discrete = is_Discrete
    self.string_to_num = {}
    self.num_to_string = {}
  
  def isString(self):
    return self.is_String
  
  def isDiscrete(self):
    return self.is_Discrete

  def numToString(self, num):
    return self.num_to_string[num]

  def stringToNum(self, string):
    return self.string_to_num[string]
  
  def addStringAndNum(self, string, num):
    self.string_to_num[string] = num
    self.num_to_string[num] = string
    

def createColumnDict(df):
  columns = {}
  for column in df:
    isString = df[column].dtype == 'object'
    isDiscrete = df[column].nunique() <= DISCRETE_THRESHOLD
    newColumn = Column(isString, isDiscrete)
    columns[column] = newColumn

  return columns


def main():

  data_name = "data/Employee-Attrition.csv"
  output_name = "output/out.csv"

  imported_df = pd.read_csv(data_name)
  
  # creates a map of columns with key = column name,
  # value is column object containing data about the column
  columns = createColumnDict(imported_df)
  
  # converts the columns of strings to ints and stores the map in column object
  for col in columns:
    # print(col, columns[col].isString(), columns[col].isDiscrete())
    if (columns[col].isString()):
      print(col)
      
  imported_df.to_csv(output_name, index=False)
  
  c = Column(True, True)
  c.addStringAndNum("Yes", 1)
  print(c.stringToNum("Yes"))
  print(c.numToString(1))
  

    
if __name__ == '__main__':
  main()