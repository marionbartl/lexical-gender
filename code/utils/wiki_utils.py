import re


def parse_wiki(raw_content):
    """
    code to parse wikipedia article into raw text
    adapted from user osantana's answer at
    https://stackoverflow.com/questions/37605045/cleaning-wikipedia-content-with-python"""

    section_title_re = re.compile("^=+\s+.*\s+=+$")
    whitespace = re.compile('^\s{2,}$')

    content = []
    skip = False
    for l in raw_content.splitlines():
        line = l.strip()
        if "= references =" in line.lower():
            skip = True  # replace with break if this is the last section
            continue
        if "= further reading =" in line.lower():
            skip = True  # replace with break if this is the last section
            continue
        if section_title_re.match(line):
            skip = False
            continue
        if not line or whitespace.match(line):
            skip = False
            continue
        if skip:
            continue
        content.append(line)

    return ' '.join(content)
