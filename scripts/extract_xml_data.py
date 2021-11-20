import os
import re
import xml.etree.ElementTree as ET
import csv

DIRECTORY = 'data/profiling-hate-speech-spreaders-twitter/pan21-author-profiling-training-2021-03-14/'
REGEX = r"^.*\\([A-Za-z0-9]{1,})\."


def extract_xml_data():
    columns = ['id','text','HS']
    for current_directory in os.listdir(DIRECTORY):
        dirs = os.path.join(DIRECTORY, current_directory)
        # lines = list()
        # lines.append(columns)

        with open(dirs + "/tweets-hs-spreaders.csv", "w", encoding='utf-8') as f:
            # f.write(''.join(lines))
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(columns)
            for filename in os.listdir(dirs):
                file = os.path.join(dirs, filename)

                if file.endswith('xml'):
                    tree = ET.parse(file)
                    root = tree.getroot()
                    tweets = [child.text for child in root.findall('.//document')]
                    match = re.search(REGEX, file)
                    user_id = match.groups(1)[0]
                    hs = root.get('class')

                    for tweet in tweets:
                        row = [user_id,tweet,hs]
                        writer.writerow(row)