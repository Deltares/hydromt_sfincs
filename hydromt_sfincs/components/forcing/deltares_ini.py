# -*- coding: utf-8 -*-
"""
Created on Wed May 19 09:52:21 2021

@author: ormondt
"""
import pandas as pd


class Section:
    def __init__(self, name=None, keyword=[], data=None):
        self.name = None
        self.keyword = []
        self.data = None

    def get_value(self, keyword):
        for kw in self.keyword:
            if kw.name.lower() == keyword.lower():
                return kw.value


class Keyword:
    def __init__(self, name=None, value=None, comment=None):
        self.name = name
        self.value = value
        self.comment = comment


class IniStruct:
    def __init__(self, filename=None):
        self.section = []

        if filename:
            self.read(filename)

    def read(self, filename):
        import re

        self.section = []
        istart = []

        fid = open(filename, "r")
        lines = fid.readlines()
        fid.close()

        # First go through lines and find start of sections

        for i, line in enumerate(lines):
            ll = line.strip()
            if len(ll) == 0:
                continue
            if ll[0] == "[" and ll[-1] == "]":
                # new section
                section_name = ll[1:-1]
                sec = Section()
                sec.name = section_name
                istart.append(i)
                self.section.append(sec)

        # Now loop through sections
        for isec in range(len(self.section)):
            i1 = istart[isec] + 1
            if isec == len(self.section) - 1:
                i2 = len(lines)
            else:
                i2 = istart[isec + 1] - 1

            df = pd.DataFrame()

            # First keyword/value pairs
            for iline in range(i1, i2):
                ll = lines[iline].strip()

                if len(ll) == 0:
                    continue

                if ll[0] == "#":
                    # comment line
                    continue

                if "=" in ll:
                    # Must be key/val pair
                    key = Keyword()

                    # First find comment
                    if "#" in ll:
                        ipos = [(i.start()) for i in re.finditer("#", ll)]
                        if len(ipos) > 1:
                            # data in between first to #
                            # remove first two #
                            ll = (
                                ll[0 : ipos[0]]
                                + ll[ipos[0] + 1 : ipos[1]]
                                + ll[ipos[1] + 1 :]
                            )

                    if "#" in ll:
                        j = ll.index("#")
                        key.comment = ll[j + 1 :].strip()
                        ll = ll[0:j].strip()

                    # Now keyword and value
                    tx = ll.split("=")
                    key.name = tx[0].strip()
                    key.value = tx[1].strip()

                    self.section[isec].keyword.append(key)

                else:
                    # And now for the data
                    a_list = ll.split()
                    list_of_floats = []
                    for item in a_list:
                        try:
                            list_of_floats.append(float(item))
                        except:
                            list_of_floats.append(item)
                    a_series = pd.Series(list_of_floats)
                    df = pd.concat([df, a_series], axis=1)

            if not df.empty:
                df = df.transpose()
                df = df.set_index([0])
                self.section[isec].data = df

    def write(self, file_name):
        import datetime

        import numpy as np

        try:
            fid = open(file_name, "w")
        except:
            print("Warning! Could not create file : " + file_name)
            return

        try:
            for section in self.section:
                fid.write("[" + section.name + "]\n")

                # First the keywords
                for kw in section.keyword:
                    value = kw.value
                    valstr = kw.value

                    kwstr = "   " + kw.name.ljust(20) + " = "

                    if value is not None:
                        if isinstance(value, float):
                            valstr = str(value)
                        elif isinstance(value, int):
                            valstr = str(value)
                        elif isinstance(value, datetime.date):
                            valstr = value.strftime("%Y%m%d")
                    else:
                        valstr = ""

                    valstr = valstr.ljust(30)
                    comstr = kw.comment
                    if kw.comment is None:
                        comstr = ""
                    else:
                        comstr = " # " + kw.comment

                    s = kwstr + valstr + comstr + "\n"
                    fid.write(s)

                # And now the data (only numeric data with two columns for now)
                if section.data is not None:
                    tt = section.data.index.to_numpy()
                    vv = section.data.array.to_numpy()
                    for it, v in enumerate(vv):
                        if np.isnan(v):
                            v = -999.0
                        string = f"{tt[it]:12.1f}{v:12.3f}\n"
                        fid.write(string)

                fid.write("\n")

            fid.close()

        except:
            fid.close()

    def get_value(self, section_name, keyword_name):
        output = None
        for section in self.section:
            if section.name == section_name:
                for kw in section.keyword:
                    if kw.name == keyword_name:
                        output = kw.value
                        return output

    def set_value(self, section_name, keyword_name, keyword_value, keyword_comment):
        # First check if this section already exists
        for section in self.section:
            if section.name == section_name:
                # Check if keyword already exist
                for kw in section.keyword:
                    if kw.name == keyword_name:
                        kw.value = keyword_value
                        return
                # New keyword
                kw = Keyword()
                kw.name = keyword_name
                kw.value = keyword_value
                kw.comment = keyword_comment
                section.keyword.append(kw)
                return

        # Make a new section
        kw = Keyword()
        kw.name = keyword_name
        kw.value = keyword_value
        kw.comment = keyword_comment
        section = Section()
        section.name = section_name
        section.keyword.append(kw)
        self.section.append(section)

    def get_data(self, section_name, keyword_list=None, value_list=None):
        output = None

        for section in self.section:
            if section.name == section_name:
                okay = True

                if keyword_list and value_list:
                    for ikey, keyword in enumerate(keyword_list):
                        for kw in section.keyword:
                            if kw.name == keyword:
                                if value_list[ikey] != kw.value:
                                    okay = False
                                continue

                if okay:
                    output = section.data
                    break

        return output

    def get_section(self, section_name, keyword_list=None, value_list=None):
        output = []

        for section in self.section:
            if section.name == section_name:
                okay = True

                if keyword_list and value_list:
                    for ikey, keyword in enumerate(keyword_list):
                        for kw in section.keyword:
                            if kw.name == keyword:
                                if value_list[ikey] != kw.value:
                                    okay = False
                                continue

                if okay:
                    output = section
                    break

        return output
