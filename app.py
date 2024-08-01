import sys
import csv
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, QFileDialog, QLabel, QMessageBox, QCheckBox, QScrollArea, QHBoxLayout
import pandas as pd
import dfhelper

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
    print("Dictionary:")
    for key in self.num_to_string:
      print(key, self.num_to_string[key])
    
  def printColumn(self):
    print("String:", self.isString())
    print("Discrete:", self.isDiscrete())
    if self.isString():
      self.printDict()

class MlProject(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.csv_data = []
        self.checkboxes = {}
        self.df = None
        self.selected_df = None
        self.targetVariable = None
        self.columns = None
        
    def initUI(self):
        self.setWindowTitle('ML Project')

        self.layout = QVBoxLayout()

        self.uploadButton = QPushButton('Upload CSV File', self)
        self.uploadButton.clicked.connect(self.uploadFile)
        self.layout.addWidget(self.uploadButton)

        self.fileNameLabel = QLabel('No file uploaded', self)
        self.layout.addWidget(self.fileNameLabel)

        self.selectColumnLabel = QLabel('Target Variable:', self)
        self.layout.addWidget(self.selectColumnLabel)

        self.selectColumnDropdown = QComboBox(self)
        self.selectColumnDropdown.currentIndexChanged.connect(self.updateCheckboxes)
        self.layout.addWidget(self.selectColumnDropdown)

        # Add a label to show if the target variable is discrete
        self.targetVariableStatus = QLabel('Status: ', self)
        self.layout.addWidget(self.targetVariableStatus)

        self.columnsLabel = QLabel('Select Features:', self)
        self.layout.addWidget(self.columnsLabel)

        self.scrollArea = QScrollArea(self)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setMinimumHeight(200)
        self.layout.addWidget(self.scrollArea)

        self.selectAllButton = QPushButton('Select All', self)
        self.selectAllButton.clicked.connect(self.selectAllCheckboxes)
        self.layout.addWidget(self.selectAllButton)

        self.deselectAllButton = QPushButton('Deselect All', self)
        self.deselectAllButton.clicked.connect(self.deselectAllCheckboxes)
        self.layout.addWidget(self.deselectAllButton)

        self.submitButton = QPushButton('Submit', self)
        self.submitButton.clicked.connect(self.submit)
        self.layout.addWidget(self.submitButton)

        self.setLayout(self.layout)
        self.setGeometry(300, 300, 400, 400)

    def uploadFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            self.loadCSV(fileName)
            self.fileNameLabel.setText(f'Uploaded File: {os.path.basename(fileName)}')
         
    def loadCSV(self, fileName):
        with open(fileName, newline='', encoding='utf-8-sig') as csvfile:
            csvreader = csv.reader(csvfile)
            self.header = next(csvreader)
            self.header = [col.lstrip('\ufeff') for col in self.header]
            self.csv_data = list(csvreader)
            self.selectColumnDropdown.clear()
            self.selectColumnDropdown.addItems(self.header)
            self.loadCheckboxes()

    def loadCheckboxes(self):
        for i in reversed(range(self.scrollAreaLayout.count())):
            widget_to_remove = self.scrollAreaLayout.itemAt(i).widget()
            self.scrollAreaLayout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)
        
        self.checkboxes = {}
        for column in self.header:
            checkbox = QCheckBox(column)
            checkbox.setChecked(True)
            self.checkboxes[column] = checkbox
            self.scrollAreaLayout.addWidget(checkbox)
        
        self.updateCheckboxes()

    def updateCheckboxes(self):
        self.target_variable = self.selectColumnDropdown.currentText()
        for column, checkbox in self.checkboxes.items():
            if column == self.target_variable:
                checkbox.setChecked(True)
                checkbox.setDisabled(True)
            else:
                checkbox.setDisabled(False)
        
        self.updateTargetVariableStatus()
        
    def get_column_data_from_csv(self, column_name):
        if column_name in self.header:
            column_index = self.header.index(column_name)
            # Extract column data from self.csv_data
            column_data = [row[column_index] for row in self.csv_data]
            return column_data
        else:
            raise ValueError(f"Column name '{column_name}' not found in CSV header.")
        
    def updateTargetVariableStatus(self):
        if self.csv_data:
            try:
                data = self.get_column_data_from_csv(self.target_variable)
                target_series = pd.Series(data)

                # Check if the target variable is discrete
                is_discrete = len(target_series.unique()) <= dfhelper.DISCRETE_THRESHOLD
                status = 'Discrete' if is_discrete else 'Continuous'
            except ValueError:
                # Handle case where target variable is not found in header
                status = 'Unknown'
        else:
            status = ' '
        self.targetVariableStatus.setText(f'Status: {status}')
            
    def selectAllCheckboxes(self):
        for column, checkbox in self.checkboxes.items():
            if checkbox.isEnabled():
                checkbox.setChecked(True)

    def deselectAllCheckboxes(self):
        for column, checkbox in self.checkboxes.items():
            if checkbox.isEnabled():
                checkbox.setChecked(False)

    def submit(self):
        
        self.targetVariable = self.selectColumnDropdown.currentText()
        if not self.targetVariable:
            QMessageBox.warning(self, "No Selection", "Please select a special column from the dropdown list.")
            return

        selected_features = [col for col, checkbox in self.checkboxes.items() if checkbox.isChecked()]
        
        if self.targetVariable not in selected_features:
            selected_features.append(self.targetVariable)
        
        for col in selected_features:
            if col != self.targetVariable:
                print(col)
      
        # selects only the selected_features from the csv_data and creates a new dataframe
        if self.csv_data:
            self.df = pd.DataFrame(self.csv_data, columns=self.header)
            self.selected_df = self.df[selected_features].copy()  # Use .copy() to avoid chained assignment
            
            for col in self.selected_df.columns:
                self.selected_df[col] = pd.to_numeric(self.selected_df[col], errors='ignore')

            
            self.columns = dfhelper.createColumnDict(self.selected_df)
            dfhelper.convertStringToInt(self.selected_df, self.columns)
            dfhelper.printColumns(self.columns)

            # output the selected data as a csv
            self.selected_df.to_csv("output/out.csv", index=False)

        else:
            print("No csv data found")
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MlProject()
    ex.show()
    sys.exit(app.exec_())
