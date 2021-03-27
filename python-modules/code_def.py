#----------------------------------------------------------------------
# Copyright 2012 EPCC, The University of Edinburgh
#
# This file is part of archer-monitoring.
#
# archer-monitoring is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# archer-monitoring is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with archer-monitoring.  If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------
"""
A python class to represent a simulation code

This class is part of the archer-monitoring framework.
The properties of the class are read from a configuration 
file that uses the [ConfigParser] module.
"""
__author__ = "A. R. Turner, EPCC"

class CodeDef(object):
    def __init__(self):
        """The default constructor - setup an simulation code system"""
        self.__name = None
        self.__regexp = None

        self.__pri_lang = None
        self.__pri_lang_version = None
        self.__aca_license = None
        self.__com_license = None
        self.__type = None
        self.__area = None

    # Properties ==============================================================
    # Code info
    @property
    def name(self):
        """The simulation code name"""
        return self.__name
    @property
    def regexp(self):
        """The regular expression used to identify the code"""
        return self.__regexp
    # Metadata
    @property
    def pri_lang(self):
        """The primary language used in the code"""
        return self.__pri_lang
    @property
    def pri_lang_version(self):
        """The version of the primary language used in the code"""
        return self.__pri_lang_version
    @property
    def aca_licence(self):
        """The academic licensing model"""
        return self.__aca_license
    @property
    def com_licence(self):
        """The commercial licensing model"""
        return self.__com_license
    @property
    def type(self):
        """The type of code"""
        return self.__type
    @property
    def area(self):
        """The research area of code"""
        return self.__area

    # Methods ==============================================================
    def readConfig(self, fileName):
        """Read the code properties from a configuration file that uses the 
        ConfigParser module.

        Arguments:
           str  fileName  - The file to read the code configuration from
        """
        import configparser

        # Set up the config for this object
        codeConfig = configparser.ConfigParser()
        codeConfig.read(fileName)

        # Get the batch information options
        self.__name = codeConfig.get("code info", "name")
        self.__regexp = codeConfig.get("code info", "regexp")

        self.__pri_lang = codeConfig.get("metadata", "primary language")
        self.__pri_lang_version = codeConfig.get("metadata", "version")
        self.__aca_license = codeConfig.get("metadata", "academic license")
        self.__com_license = codeConfig.get("metadata", "commercial license")
        self.__type = codeConfig.get("metadata", "code type")
        self.__area = codeConfig.get("metadata", "research area")

    def summaryString(self):
        """Return a string summarising the code.

           Return:
              str  output  - The string summarising the code
        """
        return "*{0}:* {1}".format(self.name,self.regexp)
