"""
A Python script to change the text in a markdown file.
The purpose is to add button html code to a specific column in a table.
The script can be run as a command line script or ran directly as a script.
"""

import argparse
import os
import sys

def main(path = None, safe = True):
    """Main function."""
    #Editable by user
    button = '<button class="md-button md-button--hyperlink">HYPERLINK</button>'
    path, safe = parse_args(path)
    lines = read_markdown_file(path)
    newlines = process_lines(lines, button)
    save_markdown_file(path, newlines, safe = safe)

def get_arguments():
    """Get the path to the markdown file and safe choice."""
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to the markdown file')
    parser.add_argument('-s', '--safe', action='store_true', help='save to a new file')
    args = parser.parse_args()
    return args.path, args.safe

def parse_args(path):
    """Get the path to the markdown file."""
    if len(sys.argv) > 1:
        path, safe = get_arguments()
    elif path is None:
        path = input('Enter the path to the markdown file: ')
    path = check_path(path)
    return path, safe

def check_path(path):
    """Check if the path exists."""
    while not os.path.exists(path):
        path = input(f'Failed to find the file at {path}. ' \
                    'Enter the path to the markdown file: ')
        if path == '':
            sys.exit()
    return path

def read_markdown_file(path):
    """Read the markdown file."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def is_table_row(line):
    """Check if the line is a table row."""
    return line.startswith('|') and not exclude_row(line)

def exclude_row(line):
    """Check if the line is a table header, the final row or already includes the button."""
    if 'Code' in line[:8] or '--' in line[:6] or '999' in line[:17]:
        return True
    if 'button' in line[:80]:
        return True
    return False

def add_button(line, button):
    """Add a button at the end of the text in the first table cell."""
    cells = line.split('|')
    cells[1] = f"{cells[1]}{button} "
    return '|'.join(cells)

def process_lines(lines, text):
    """Process the lines from a text file."""
    for i, line in enumerate(lines):
        if is_table_row(line):
            lines[i] = add_button(line, text)
    return lines

def save_markdown_file(path, lines, safe):
    """Save the lines to a markdown file."""
    if safe:
        path = path.replace('.md', '_new.md')
    else:
        decision = input(f'Warning! This will overwrite the file at {path}. '\
                        'Do you wish to continue? (y/n): ')
        if decision.lower() != 'y':
            return
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def test(inputpath, testpath, safe):
    """Test the script by running main on a test file and comparing the output to a reference file."""
    main(path = inputpath, safe = safe)
    if safe:
        inputpath = inputpath.replace('.md', '_new.md')
    with open(inputpath, 'r') as f:
        inputlines = f.readlines()
    with open(testpath, 'r') as f:
        testlines = f.readlines()
    assert inputlines == testlines

if __name__ == '__main__':
    main()
    #test(R".\tests\inputTest.md", R".\tests\outputTest.md", False)