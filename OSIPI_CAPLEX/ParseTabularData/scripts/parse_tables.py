"""
This script is used to load in a markdown file containing tables and parse them
into dataframes which are saved as csv files.
"""
import os
import pandas as pd
from io import StringIO
import yaml
from pprint import pprint

nav = \
"""
    - Q - Quantities: quantities.md
    - M - Perfusion models: perfusionModels.md
    - P - Perfusion processes: perfusionProcesses.md
    - G - General purpose processes: generalPurposeProcesses.md
    - D - Derived processes: derivedProcesses.md
"""

#path to the folder containing the website files
path = R"C:\Users\mbcxamk2\OneDrive - The University of Manchester\Uni\OSIPI_CAPLEX\github_repos\OSIPI_CAPLEX_MartinK"

def get_filenames(nav):
    """
    This function takes in a string containing the navigation bar and returns
    a list of the filenames.
    """
    filenames = []
    for line in nav.splitlines():
        if line.strip().startswith("-"):
            filenames.append(line.split(":")[1].strip())
    return filenames

def make_filepaths(path, filenames):
    """
    This function takes in a path and a list of filenames and returns a list of
    filepaths.
    """
    filepaths = []
    for filename in filenames:
        filepaths.append(os.path.join(path, 'docs', filename))
    return filepaths

def get_tables(filepaths):
    """
    This function takes in a list of filepaths and returns a list of tables.
    """
    tables = []
    for filepath in filepaths:
        table = find_tables(filepath)
        if table:
            tables.append([os.path.basename(filepath), table])
        else:
            print(f"No tables found in {filepath}")
    return tables

def find_tables(filepath):
    """
    This function takes in a filepath and returns a list of tables.
    """
    tables = []
    title = None
    temp = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            #get title of a table
            if line.startswith('#'):
                if '<a' in line:
                    title = line.split('a>')[-1].strip()
                else:
                    title = line.strip('#').strip()
                
                #if we find a title and a table, we save it
                if title and temp:
                    tables.append([title, temp])
                    temp = []
                    title = None
            #get the table
            if line.startswith('|'):
                #some equations have | in them, so we replace them with something else
                if R'\left\|' in line:
                    line = line.replace(R'\left\|', 'REPLACEWITHLEFTBRACKET')
                if R'\right\|' in line:
                    line = line.replace(R'\right\|', 'REPLACEWITHRIGHTBRACKET')
                temp.append(line)
        #if we reach the end of the file and there is a title and a table, we save it
        else:
            if title and temp:
                tables.append([title, temp])
    return tables

def from_tables_to_dfs(tables):
    for name, file in tables:
        for table in file:
            table[1] = pd.read_csv(StringIO(''.join(table[1])), sep='|')
            table[1].drop(labels=[col for col in table[1].columns if 'Unnamed' in col], axis=1, inplace=True)
            table[1].columns = [col.strip() for col in table[1].columns]
            table[1]['Description'] = table[1]['Description'].str.replace('REPLACEWITHLEFTBRACKET', R'\left\|')
            table[1]['Description'] = table[1]['Description'].str.replace('REPLACEWITHRIGHTBRACKET', R'\right\|')

def main(path, nav):
    filenames = get_filenames(nav)
    filepaths = make_filepaths(path, filenames)
    tables = get_tables(filepaths)
    from_tables_to_dfs(tables)
    for name, file in tables:
        for table in file:
            if "/" in table[0]:
                table[0] = table[0].replace("/", "_")
            if "<" in table[0]:
                table[0] = table[0].replace("<", "_")
            if ">" in table[0]:
                table[0] = table[0].replace(">", "_")
            if "*" in table[0]:
                table[0] = table[0].replace("*", "_")
            table[1].to_csv(
                os.path.join(
                    R'C:\Users\mbcxamk2\OneDrive - The University of Manchester\Uni\PhD project\PublicScripts\OSIPI_CAPLEX\ParseTabularData\tables', 
                    name.replace('.', '-') + table[0] + '.csv'),
                index=False
                )

if __name__ == "__main__":
    main(path, nav)
