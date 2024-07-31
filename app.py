import sys
import csv
import os
import dataframemaker
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, QFileDialog, QLabel, QMessageBox, QCheckBox, QScrollArea, QHBoxLayout, QWidget
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

        self.columnsLabel = QLabel('Select Features:', self)
        self.layout.addWidget(self.columnsLabel)

        self.scrollArea = QScrollArea(self)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setMinimumHeight(200)
        self.layout.addWidget(self.scrollArea)

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
        with open(fileName, newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            self.header = next(csvreader)
            self.csv_data = list(csvreader)
            self.selectColumnDropdown.clear()
            self.selectColumnDropdown.addItems(self.header)
            self.loadCheckboxes()
        return csvfile

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
        target_variable = self.selectColumnDropdown.currentText()
        for column, checkbox in self.checkboxes.items():
            if column == target_variable:
                checkbox.setChecked(True)
                checkbox.setDisabled(True)
            else:
                checkbox.setDisabled(False)

    def submit(self):
        selected_features = self.selectColumnDropdown.currentText()
        if not selected_features:
            QMessageBox.warning(self, "No Selection", "Please select a special column from the dropdown list.")
            return

        selected_columns = [col for col, checkbox in self.checkboxes.items() if checkbox.isChecked()]
        
        if selected_features not in selected_columns:
            selected_columns.append(selected_features)
        
        print(f"Target Variable: {selected_features}")
        print("Selected Features:")
        for col in selected_columns:
            if col != selected_features:
                print(col)
      
        print(selected_columns)
        print(self.csv_data)
        self.df = pd.read_csv(self.csv_data)
    
        self.df.to_csv("output/out.csv", index=False)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MlProject()
    ex.show()
    sys.exit(app.exec_())

