# Exploratory Data Analysis - Customer Loans in Finance

# Table of Contents
1. [Introduction](#introduction)
2. [Installation Instructions](#installation-instructions)
3. [File Structure](#file-structure)
4. [License Information](#license-information)


## Introduction
This project is an Exploratory Data Analysis (EDA) of loan data. The following processes have been applied to the dataset:
1. Convert Columns to Correct Format: Changes are made to certain columns of the dataset to make later operations easier.
2. Handling NULL Values: Any NULL values are removed, either by dropping the row, dropping the column, or imputing the values from existing data.
3. Skew Correction: Transformations are performed on certain overly-skewed columns, to create a more normal distribution of values.
4. Handling Outliers: All columns are examined for outliers, which are then removed from the dataset.
5. Dropping Co-linear Columns: Combinations of columns are evaluated to determine how highly-correlated they are, and redundant columns are removed from the dataset.
6. Analysis and Visualisation: Finally, the resulting dataset is analysed to determine profits and losses.

## Installation Instructions
You may download the contents of the repository from GitHub by either:
- Using `git clone https://github.com/WillBoyer/exploratory-data-analysis---customer-loans-in-finance521.git`
- Using the 'Download ZIP' button on the repository page.

After downloading the files, navigate to the repository's root directory and open `python data_exploration.ipynb` in Visual Studio Code.
Then, click 'Run All' in the top bar of the `python data_exploration.ipynb` window to view all code output.

## File Structure
.
├── `README.md`: The file you are currently reading. Gives explanations on the project, installation method, and licensing.
├── `data_exploration.ipynb`: The Python notebook file that performs transformation and analysis on the dataset.
└── `db_utils.py`: Contains a definition of the `RDSDatabaseConnector` class

## License Information
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.