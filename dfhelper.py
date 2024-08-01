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
  
  # adds a string-num and num-string conversion to the column
  def addStringNum(self, string, num):
    self.string_to_num[string] = num
    self.num_to_string[num] = string
    
  def checkString(self, string):
    return string in self.string_to_num
  
  def printDict(self):
    for key in self.num_to_string:
      print(key, self.num_to_string[key])
      
  def getDict(self):
    out = ""
    for key in self.num_to_string:
      out = out + "\n" + str(key) + " " + self.num_to_string[key]
    return out
      
    
  def printColumn(self):
    print("String:", self.isString())
    print("Discrete:", self.isDiscrete())
    if self.isString():
      print("Dictionary:")
      self.printDict()


# create all the column objects based on data in dataframe
def createColumnDict(df):
  columns = {}
  for column in df:
    isString = df[column].dtype == 'object'
    isDiscrete = df[column].nunique() <= DISCRETE_THRESHOLD
    newColumn = Column(isString, isDiscrete)
    columns[column] = newColumn

  return columns

def outputDictionaries(columns):
    with open('output/dictout.txt', 'w') as f:
        for column in columns:
            if columns[column].isString():
                f.write(f"{column}")
                f.write(f"{columns[column].getDict()}\n")
                f.write("\n")

def printColumns(columns):
  for column in columns:
    print(column)
    columns[column].printColumn()
    print("--")

# converts all strings of a dataframe to integers and stores add the dictionary entry to the column object
def convertStringToInt(df, columns):
  # itterate through all column names in columns dictionary
  for col in columns:
    # if the column is a string, convery entry to a mapped integer
    if (columns[col].isString()):
      current_index = 0
      for i, row in enumerate(df[col]):
        # check if string is in map, if not add a new entry
        if not columns[col].checkString(row):
          columns[col].addStringNum(row, current_index)
          current_index = current_index + 1
        
        # change the entry to the mapped value
        df.loc[i, col] = columns[col].stringToNum(row)

def main():
  data_name = "data/Employee-Attrition-test.csv"
  output_name = "output/out.csv"

  imported_df = pd.read_csv(data_name)
  
  # creates a map of columns with key = column name,
  # value is a column object containing data about the column
  columns = createColumnDict(imported_df)
  
  # converts the columns of strings to integers and stores the map in column object
  convertStringToInt(imported_df, columns)

  # printColumns(columns)
  
    
if __name__ == '__main__':
  main()